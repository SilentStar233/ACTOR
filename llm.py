# LLM infrastructure: API calls, disk cache, JSON repair-retry, token/cost logging.
import hashlib
import json
import os
import pickle
import random
import re
import threading
import time

import openai


class _EmptyResponse(Exception):
    """API returned a completion with empty/None message content."""


class LLM:
    def __init__(self, config_path='config.json',
                 cache_path='data/logs/cache.pkl',
                 log_path='data/logs/cost.jsonl',
                 default_model='gpt-5.6-luna',
                 max_rpm=None, max_tpm=None, max_retries=3,
                 book=None, run_dir=None):
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        self.client = openai.OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url'],
            timeout=300,
        )
        self.default_model = default_model
        self._lock = threading.Lock()
        self.last_finish_reason = None
        # Env defaults let the batch scheduler inject per-process budgets.
        max_rpm = max_rpm or int(os.environ.get('AT_MAX_RPM') or 0) or None
        max_tpm = max_tpm or int(os.environ.get('AT_MAX_TPM') or 0) or None
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.max_retries = max_retries
        # Stall guard: raise only when the limiter has granted nothing at
        # all for max_rate_wait (genuine stall, e.g. broken accounting).
        # A thread merely out-competed by its siblings keeps waiting: the
        # spin-preemptive buckets have no FIFO fairness, so own-wait alone
        # must not fail the book. AT_MAX_RATE_WAIT overrides the default.
        self.max_rate_wait = float(
            os.environ.get('AT_MAX_RATE_WAIT') or 1800)
        self._req_bucket = float(max_rpm or 0)
        self._tok_bucket = float(max_tpm or 0)
        self._bucket_clock = time.time()
        self._last_acquire = self._bucket_clock
        self._contention_warned = False
        self.book = book
        if book is None and os.environ.get('AT_RUN_DIR'):
            # Batch mode: falling back to the shared pickle cache would
            # corrupt it from multiple processes. Fail loudly instead.
            raise ValueError(
                'LLM(book=None) is not allowed when AT_RUN_DIR is set')
        if book is not None:
            # Per-book isolation: cache/<sha>.json files (lazy, no bulk load)
            # and cost.jsonl under run_dir/<book>/. run_dir defaults to the
            # AT_RUN_DIR env var so scripts work from any cwd.
            from utils import safe_title
            run_dir = run_dir or os.environ.get('AT_RUN_DIR', 'data/logs')
            book_dir = os.path.join(run_dir, safe_title(book))
            self.cache_dir = os.path.join(book_dir, 'cache')
            self.log_path = os.path.join(book_dir, 'cost.jsonl')
            self.cache_path = None
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cache = {}  # session-local hits only
        else:
            self.cache_dir = None
            self.cache_path = cache_path
            self.log_path = log_path
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self._cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(self._cache, f)
        os.replace(tmp, self.cache_path)

    def _cache_get(self, key):
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self.cache_dir is None:
            return None
        path = os.path.join(self.cache_dir, key + '.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding='utf-8') as f:
                text = json.load(f)['text']
        except Exception:
            return None  # corrupt entry: treat as miss, rewritten on success
        with self._lock:
            self._cache[key] = text
        return text

    def _cache_put(self, key, text):
        if self.cache_dir is not None:
            with self._lock:
                self._cache[key] = text
            path = os.path.join(self.cache_dir, key + '.json')
            tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'text': text}, f, ensure_ascii=False)
            os.replace(tmp, path)
        else:
            with self._lock:
                self._cache[key] = text
                self._save_cache()

    @staticmethod
    def _key(model, prompt, json_mode=False, max_tokens=None):
        # 'v2' salt: v1 cache entries may contain truncated responses (poisoned).
        mode = 'json' if json_mode else 'text'
        return hashlib.sha256(
            f'v2||{model}||{mode}||{max_tokens}||{prompt}'.encode('utf-8')
        ).hexdigest()

    def _acquire_rate(self, est_tokens):
        """Block until both token buckets allow this call (cached calls skip)."""
        if not self.max_rpm and not self.max_tpm:
            return
        if self.max_tpm:
            # A single call must never exceed bucket capacity, otherwise
            # tok_ok can never become true and this loop never exits.
            est_tokens = min(float(est_tokens), float(self.max_tpm))
            if (not self._contention_warned
                    and est_tokens > self.max_tpm / 3.0):
                # Structural feasibility, checked once at the first big
                # call: below ~3x headroom the spin-preemptive buckets let
                # small requests repeatedly starve big ones (no FIFO).
                # Surface the config smell immediately, at startup, rather
                # than killing the book mid-run.
                self._contention_warned = True
                print(f'  rate-limiter warning: one call needs '
                      f'~{est_tokens:.0f} tok vs max_tpm={self.max_tpm} '
                      '(<3x headroom); under concurrency expect long waits: '
                      'lower --parallel-books or raise --max-tpm',
                      flush=True)
        waited = 0.0
        next_report = 60.0
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._bucket_clock
                self._bucket_clock = now
                if self.max_rpm:
                    self._req_bucket = min(
                        float(self.max_rpm),
                        self._req_bucket + elapsed * self.max_rpm / 60.0)
                if self.max_tpm:
                    self._tok_bucket = min(
                        float(self.max_tpm),
                        self._tok_bucket + elapsed * self.max_tpm / 60.0)
                req_ok = not self.max_rpm or self._req_bucket >= 1.0
                tok_ok = not self.max_tpm or self._tok_bucket >= est_tokens
                if req_ok and tok_ok:
                    if self.max_rpm:
                        self._req_bucket -= 1.0
                    if self.max_tpm:
                        self._tok_bucket -= est_tokens
                    self._last_acquire = now
                    return
                wait = 1.0
                if self.max_rpm and not req_ok:
                    wait = max(wait, (1.0 - self._req_bucket) * 60.0 / self.max_rpm)
                if self.max_tpm and not tok_ok:
                    wait = max(wait, (est_tokens - self._tok_bucket) * 60.0 / self.max_tpm)
            slept = min(wait, 30.0)
            time.sleep(slept)
            waited += slept
            if waited > self.max_rate_wait:
                with self._lock:
                    idle = time.time() - self._last_acquire
                if idle > self.max_rate_wait:
                    raise RuntimeError(
                        f'rate limiter stalled: nothing granted for '
                        f'{idle:.0f}s (this call needs {est_tokens:.0f} tok, '
                        f'max_tpm={self.max_tpm}, max_rpm={self.max_rpm}); '
                        'lower --parallel-books or raise --max-tpm/--max-rpm')
                # Sibling threads are still acquiring: contended, not
                # stalled. Keep waiting rather than failing the book.
            # Long waits must stay observable (stdout lands in pipeline.log,
            # which also feeds the batch stall detector's mtime watch):
            # exactly one line per 60s of waiting, never a burst.
            if waited >= next_report:
                print(f'  rate-limited: waited {waited:.0f}s '
                      f'(need {est_tokens:.0f} tok, '
                      f'bucket {self._tok_bucket:.0f})', flush=True)
                next_report = waited + 60.0

    @staticmethod
    def _retryable(error):
        if isinstance(error, (_EmptyResponse, openai.RateLimitError,
                              openai.APIConnectionError, openai.APITimeoutError)):
            return True
        status = getattr(error, 'status_code', None)
        return status is not None and status >= 500

    @staticmethod
    def _retry_wait(error, attempt):
        response = getattr(error, 'response', None)
        if response is not None:
            retry_after = response.headers.get('retry-after')
            if retry_after:
                try:
                    return min(float(retry_after), 120.0)
                except ValueError:
                    pass
        # Jitter: N concurrent workers must not retry in lockstep.
        return min(2.0 * (2 ** attempt), 60.0) * random.uniform(0.6, 1.4)

    def _log(self, record):
        record['time'] = time.time()
        if self.book is not None:
            record.setdefault('book', self.book)
        with self._lock:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def call(self, prompt, model=None, temperature=0.0, max_tokens=16384,
             use_cache=True, json_mode=False):
        """Call the LLM, return raw text. Cached by (model, json_mode, prompt)."""
        model = model or self.default_model
        key = self._key(model, prompt, json_mode, max_tokens)
        if use_cache:
            cached_text = self._cache_get(key)
            if cached_text is not None:
                self._log({'model': model, 'cached': True, 'input_chars': len(prompt)})
                return cached_text

        # gpt-family (e.g. gpt-5.6-luna) rejects max_tokens/temperature=0:
        # it requires max_completion_tokens and its own default temperature.
        request = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if model.startswith(('gpt-', 'o1', 'o3', 'o4')):
            request['max_completion_tokens'] = max_tokens
            if temperature:
                request['temperature'] = temperature
        else:
            request['max_tokens'] = max_tokens
            request['temperature'] = temperature
        if json_mode:
            request['response_format'] = {'type': 'json_object'}

        # Estimate with an empirical output cap (measured avg completion is
        # ~1.6k tokens; max_tokens is a hard ceiling, not an expectation),
        # then reconcile the bucket with real usage after the response.
        est_tokens = len(prompt) // 4 + min(max_tokens, 4096)
        completion = None
        text = None
        for attempt in range(self.max_retries + 1):
            # Charge every attempt: retried calls really hit the upstream
            # (rate accounting matters most exactly when 429s happen).
            self._acquire_rate(est_tokens)
            try:
                completion = self.client.chat.completions.create(**request)
                if not completion.choices:
                    raise _EmptyResponse('no choices in completion')
                text = completion.choices[0].message.content
                if text:
                    break
                # Empty content (content filter, upstream fluke, reasoning-only
                # response): retry like a transient server error.
                raise _EmptyResponse('empty message content')
            except Exception as error:
                if attempt >= self.max_retries or not self._retryable(error):
                    raise
                wait = self._retry_wait(error, attempt)
                print(f'  llm retry {attempt + 1}/{self.max_retries} '
                      f'after {type(error).__name__}, sleeping {wait:.0f}s')
                time.sleep(wait)
        finish_reason = getattr(completion.choices[0], 'finish_reason', None)
        self.last_finish_reason = finish_reason
        usage = getattr(completion, 'usage', None)
        if self.max_tpm and usage is not None:
            # Reconcile the token bucket with real usage: over-estimates are
            # refunded, under-estimates charged (bucket may go negative =
            # borrowing from the next window).
            actual = ((getattr(usage, 'prompt_tokens', 0) or 0)
                      + (getattr(usage, 'completion_tokens', 0) or 0))
            charged = min(float(est_tokens), float(self.max_tpm))
            with self._lock:
                self._tok_bucket -= (actual - charged)
        self._log({
            'model': model,
            'cached': False,
            'input_chars': len(prompt),
            'input_tokens': getattr(usage, 'prompt_tokens', None),
            'output_tokens': getattr(usage, 'completion_tokens', None),
            'finish_reason': finish_reason,
        })

        # Truncated responses are never cached: they are partial and would
        # poison every future run that hits the same key.
        if use_cache and text and finish_reason != 'length':
            self._cache_put(key, text)
        return text

    @staticmethod
    def parse_json(text):
        """Best-effort JSON parsing: strip markdown fences, take outermost braces."""
        text = text.strip()
        m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError('No JSON object found in response')
        return json.loads(text[start:end + 1])

    def call_json(self, prompt, model=None, retries=3, **kwargs):
        """Call the LLM and parse JSON.

        - Truncated responses (finish_reason='length') raise immediately: a repair
          pass cannot recover content that was never generated.
        - Parse failures on complete responses trigger a repair retry.
        """
        model = model or self.default_model
        self.last_finish_reason = None
        response = self.call(prompt, model=model, **kwargs)
        if getattr(self, 'last_finish_reason', None) == 'length':
            raise ValueError('Response truncated (finish_reason=length)')
        last_err = None
        for _ in range(retries):
            try:
                return self.parse_json(response)
            except Exception as e:
                last_err = e
                repair_prompt = (
                    'The following text was supposed to be a single valid JSON object but failed '
                    'to parse. Fix it and output ONLY the corrected JSON: no markdown fences, '
                    'no commentary, no truncation.\n\n===Broken text===\n' + response
                )
                response = self.call(repair_prompt, model=model,
                                     **{**kwargs, 'use_cache': False})
                if getattr(self, 'last_finish_reason', None) == 'length':
                    raise ValueError('Repair response truncated (finish_reason=length)')
        raise ValueError(f'JSON parsing failed after {retries} attempts: {last_err}')

    def cost_summary(self):
        """Aggregate token usage from the log file."""
        total_in, total_out, calls, cached = 0, 0, 0, 0
        if not os.path.exists(self.log_path):
            return {'calls': 0}
        with open(self.log_path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('cached'):
                    cached += 1
                else:
                    calls += 1
                    total_in += r.get('input_tokens') or 0
                    total_out += r.get('output_tokens') or 0
        return {
            'api_calls': calls,
            'cached_calls': cached,
            'input_tokens': total_in,
            'output_tokens': total_out,
        }
