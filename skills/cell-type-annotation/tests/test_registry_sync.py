"""Keep the docs, the CLI, and the package in sync.

The `cta` command surface (`cta.cli.GROUPS` → `cta.cli.command_paths()`) is the single
source of truth. These tests fail if `reference/tool_registry.md` lists a command the CLI
doesn't have (or vice versa), if SKILL.md's command table misses a command, or if a group
entry isn't a runnable command — so the docs can't silently drift from the code.
"""
from __future__ import annotations

import pathlib
import re

import typer

from cta.cli import GROUPS, command_paths

_SKILL_DIR = pathlib.Path(__file__).resolve().parent.parent
_REGISTRY = _SKILL_DIR / "reference" / "tool_registry.md"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _command_tokens(path: pathlib.Path) -> list[str]:
    """Backtick-wrapped `cta <…>` tokens in a doc (e.g. `cta markers compute`).

    Tokens may carry trailing flags (`cta report build --strict`); callers match by
    prefix where that matters.
    """
    return [m.strip() for m in re.findall(r"`cta ([a-z0-9][a-z0-9 -]*?)`", path.read_text())]


def _documents_path(tokens: list[str], p: str) -> bool:
    """True if some backtick token is the command path `p` (bare or with trailing flags)."""
    return any(t == p or t.startswith(p + " ") for t in tokens)


def test_every_command_is_documented():
    tokens = set(_command_tokens(_REGISTRY))
    missing = [p for p in command_paths() if p not in tokens]
    assert not missing, f"commands absent from reference/tool_registry.md: {missing}"


def test_registry_has_no_unknown_commands():
    paths = command_paths()
    unknown = [t for t in _command_tokens(_REGISTRY)
               if not any(t == p or t.startswith(p + " ") for p in paths)]
    assert not unknown, f"tool_registry.md references unknown cta commands: {unknown}"


def test_every_command_in_skill_md():
    """SKILL.md's Commands table must mention every command (bare or with flags)."""
    tokens = _command_tokens(_SKILL_MD)
    missing = [p for p in command_paths() if not _documents_path(tokens, p)]
    assert not missing, f"commands absent from SKILL.md: {missing}"


def test_group_entries_are_runnable():
    for group, commands in GROUPS.items():
        for name, cmd in commands.items():
            assert callable(cmd) or isinstance(cmd, typer.Typer), \
                f"{group}/{name} is neither a command function nor a Typer app"


def test_every_command_help_builds():
    """`cta <path> --help` must exit 0 for every command — catches a broken option
    surface (e.g. a malformed shared Option) across all commands, not just the smoke-tested
    ones. --help doesn't run the body, so this stays fast (no heavy imports)."""
    import subprocess
    for path in command_paths():
        r = subprocess.run(["cta", *path.split(), "--help"], capture_output=True, text=True)
        assert r.returncode == 0, f"`cta {path} --help` failed:\n{r.stderr}"
