"""The rendered run prompt carries the IR-extension workflow and a hook-safe bench command."""

from pathlib import Path

import pytest

from evolution.cli import REPO_ROOT, _prompt
from evolution.preparation.declare import (
    SotaBaseline,
    Task,
    load_task,
    load_toolset,
    task_path,
    toolset_path,
)
from evolution.prompts.render import kernel_ir_extension_contract

WORKTREE = Path("/runs/example/worktree")


def _task(name: str = "kda_backward") -> Task:
    return load_task(task_path(REPO_ROOT, name))


def _toolset():
    return load_toolset(toolset_path(REPO_ROOT, "kda_flow"))


def test_tirx_lite_prompt_describes_the_extension_workflow():
    text = _prompt(
        task=_task(), toolset=_toolset(), worktree=WORKTREE, python=WORKTREE / ".venv/bin/python"
    )
    section = text.split("### IR extensions", 1)[1].split("## 7.", 1)[0]
    assert "tirx_harness.extension.lower(kernel, *passes)" in section
    assert "references/tirx-ir-extension.md" in section
    assert "synccheck/racecheck/NumSim" in section and ".compile()" in section
    assert "check_ir=True" in section
    # The extension block sits inside the implementation contract, before the frontier rules.
    assert text.index("## 6. Implementation contract") < text.index("### IR extensions")
    assert "{{" not in text


def test_task_authoring_mode_omits_the_extension_block():
    task = Task(
        name="other",
        workload_dir="candidates/other",
        kernel_authoring="task",
        sota_baseline=SotaBaseline(name="baseline"),
        spec="Spec.",
    )
    assert kernel_ir_extension_contract(task) == ""
    text = _prompt(task=task, toolset=_toolset(), worktree=WORKTREE)
    assert "IR extensions" not in text and "{{" not in text


def test_bench_command_uses_the_worktree_relative_interpreter():
    text = _prompt(
        task=_task(), toolset=_toolset(), worktree=WORKTREE, python=WORKTREE / ".venv/bin/python"
    )
    assert "timeout 300s .venv/bin/python evolution/benchmark/adapter.py" in text
    assert (
        str(WORKTREE / ".venv") not in text.split("## 4. Benchmark protocol")[1].split("## 5.")[0]
    )


def test_interpreter_outside_the_worktree_stays_absolute():
    text = _prompt(
        task=_task(), toolset=_toolset(), worktree=WORKTREE, python=Path("/opt/python/bin/python3")
    )
    assert "timeout 300s /opt/python/bin/python3 evolution/benchmark/adapter.py" in text


@pytest.mark.parametrize("remote", ["http://127.0.0.1:61886"])
def test_remote_prompt_keeps_the_extension_block(remote):
    text = _prompt(
        task=_task(),
        toolset=_toolset(),
        worktree=WORKTREE,
        remote_url=remote,
        python=WORKTREE / ".venv/bin/python",
    )
    assert "### IR extensions" in text
    assert (
        f".venv/bin/python evolution/remote/kcoral_remote.py candidates/kda/backward CANDIDATE --remote {remote}"
        in text
    )
