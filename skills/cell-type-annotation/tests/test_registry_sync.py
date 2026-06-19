"""Keep the docs, the CLI, and the package in sync.

The `cta` command surface (`cta.cli.GROUPS` → `cta.cli.command_paths()`) is the single
source of truth. These tests fail if `reference/tool_registry.md` lists a command the CLI
doesn't have (or vice versa), or if a group entry isn't a runnable command — so the
canonical tool table can't silently drift from the code.
"""
from __future__ import annotations

import pathlib
import re

import typer

from cta.cli import GROUPS, command_paths

_SKILL_DIR = pathlib.Path(__file__).resolve().parent.parent
_REGISTRY = _SKILL_DIR / "reference" / "tool_registry.md"


def _registry_command_tokens() -> list[str]:
    """Backtick-wrapped `cta <…>` tokens in the registry (e.g. `cta markers compute`)."""
    text = _REGISTRY.read_text()
    return [m.strip() for m in re.findall(r"`cta ([a-z0-9][a-z0-9 -]*?)`", text)]


def test_every_command_is_documented():
    tokens = set(_registry_command_tokens())
    missing = [p for p in command_paths() if p not in tokens]
    assert not missing, f"commands absent from reference/tool_registry.md: {missing}"


def test_registry_has_no_unknown_commands():
    paths = command_paths()
    unknown = [t for t in _registry_command_tokens()
               if not any(t == p or t.startswith(p + " ") for p in paths)]
    assert not unknown, f"tool_registry.md references unknown cta commands: {unknown}"


def test_group_entries_are_runnable():
    for group, commands in GROUPS.items():
        for name, cmd in commands.items():
            assert callable(cmd) or isinstance(cmd, typer.Typer), \
                f"{group}/{name} is neither a command function nor a Typer app"
