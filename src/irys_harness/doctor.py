from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def run_doctor(root: str | Path = ".") -> list[CheckResult]:
    repo_root = Path(root).resolve()
    return [
        check_env(repo_root),
        check_private_ignores(repo_root),
        check_harvey_sibling(repo_root),
    ]


def check_env(repo_root: Path) -> CheckResult:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return CheckResult(".env", False, ".env is missing")
    values = read_env_keys(env_path)
    if not values.get("GEMINI_API_KEY"):
        return CheckResult(".env", False, "GEMINI_API_KEY is missing or empty")
    return CheckResult(".env", True, "GEMINI_API_KEY is configured")


def read_env_keys(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def check_private_ignores(repo_root: Path) -> CheckResult:
    paths = [".env", ".irys-private", "scratch", "traces", "outputs", "results"]
    missing = [path for path in paths if not is_git_ignored(repo_root, path)]
    if missing:
        return CheckResult(
            "private_ignores",
            False,
            "Not ignored: " + ", ".join(missing),
        )
    return CheckResult("private_ignores", True, "private and run-artifact paths are ignored")


def is_git_ignored(repo_root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=repo_root,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def check_harvey_sibling(repo_root: Path) -> CheckResult:
    harvey = (repo_root / ".." / "harvey-labs").resolve()
    if not harvey.exists():
        return CheckResult("harvey_labs", False, f"Missing sibling repo: {harvey}")
    tasks = harvey / "tasks"
    evaluation = harvey / "evaluation"
    if not tasks.exists() or not evaluation.exists():
        return CheckResult("harvey_labs", False, "Sibling exists but tasks/evaluation dirs are missing")
    return CheckResult("harvey_labs", True, f"Found Harvey LAB sibling at {harvey}")

