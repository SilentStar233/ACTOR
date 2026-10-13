# ACTOR 案例展示（README 插入草稿）

## Benchmark 题目示例

给定角色档案、记忆和当前场景，预测该角色接下来最可能采取的行为。

> 以下为真实题目的中文摘要；完整模型输入为英文，模型只需回答 A/B/C/D。

| 输入 | 内容摘要 |
| --- | --- |
| 目标角色 | Character 1，一名管家与使者 |
| 角色档案 | 忠于女主人的阵营，将 Character 5 视为敌方人物，并愿意以暴力攻击他 |
| 当前场景 | 他准备攻击失明的 Character 5，Character 10 出面阻挡；双方已发生威胁和辱骂 |

**Character 1 接下来最可能做什么？**

| A | B | C | D |
| --- | --- | --- | --- |
| 向 Character 10 下跪 | 丢弃武器 | 与 Character 10 战斗 | 保护失明的 Character 5 |

**参考答案：C。** 这里要预测角色的行为，而不是选择一般意义上更善良的行为。

<details>
<summary>数据字段一览</summary>

| 字段 | 内容 |
| --- | --- |
| `profile` | 当前角色档案 |
| `long_term_memory` / `recent_memory` | 历史与近期的 Thought、Action、Observation |
| `scene` | 当前场景、时间及参与者描述 |
| `options` | 四个候选行为，包含 `id` 和 `text` |
| `correct_answer` | 参考答案，仅用于评分，不输入模型 |
| `question_id` / `split` / `is_critical` | 题目标识、评测分区和行为类型 |

每条题目是一个 JSONL 记录；上方卡片省略了完整记忆。标准匿名评测不展示原书名和原人物姓名。

</details>

## RPO 训练样本示例

训练使用同一角色上下文下的 **chosen / rejected 行为文本**，不是选项字母。

| 上下文摘要 | Character 1 是一名修道院院长，早餐后来拜访 Character 2，对方亲切地与她交谈 |
| --- | --- |
| chosen（原文） | Listens and smiles without saying a word while Character 2 talks of Court intrigues. |
| rejected（原文） | Hastens to unseal the register, exposing each sheltered pensioner to Character 2 imprudent curiosity. |

chosen 表示微笑聆听，rejected 则表示打开登记册、暴露受庇护者的信息。RPO 在偏好损失之外加入 chosen 回答的 NLL 项，由 `rpo_alpha` 控制权重。

## 训练前后：同一道题，选择发生了什么变化？

以下为 Qwen2.5-7B-Instruct 与 RPO checkpoint-6000 在相同匿名题目上的真实预测。

| 案例 | 基础模型 | RPO 训练后 | 参考答案 |
| --- | --- | --- | --- |
| 上方管家冲突题（《李尔王》） | D：保护失明者 ✗ | C：与阻挡者战斗 ✓ | C |
| 家庭中的疏远关系（《简·爱》） | C：主动寻找对方作伴 ✗ | B：尽量减少与对方交谈 ✓ | B |

书名仅用于标注来源，不出现在匿名提示词中。行为文字来自选项，模型实际输出的是字母；这些是精选案例，不代表所有题目都会改善。

<details>
<summary>案例来源与追溯</summary>

- 评测题 ID：`King Lear::oswald::p029_s000_t002_a004`、`Jane Eyre::eliza_reed::p007_s000_t000_a000`。
- 训练样本 ID：`The Three Musketeers::abbess::p129_s003_t001_a001`。
- 来源：20260916 正式评测产物及 20260828 重建训练数据；预测已按题目 ID 核对。案例解读不是模型生成的推理。

</details>
