"""Repository paths independent of the calling directory or module location."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = str(REPOSITORY_ROOT)
