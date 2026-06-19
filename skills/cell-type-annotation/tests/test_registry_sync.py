"""Keep the docs, the CLI, and the package in sync.

The `cta` command set (``cta.cli.COMMANDS``) is the single source of truth. These tests
fail if `reference/tool_registry.md` lists a command the CLI doesn't have (or vice versa),
or if a command points at a module that isn't in the package — so the canonical tool table
can't silently drift from the code.
"""
from __future__ import annotations

import pathlib
import re

import cta
from cta.cli import COMMANDS

_SKILL_DIR = pathlib.Path(__file__).resolve().parent.parent
_REGISTRY = _SKILL_DIR / "reference" / "tool_registry.md"


def _registry_commands() -> set[str]:
    """`cta <command>` tokens that appear backtick-wrapped in the registry table."""
    text = _REGISTRY.read_text()
    return set(re.findall(r"`cta ([a-z][a-z0-9-]+)", text))


def test_every_cli_command_is_in_the_registry():
    missing = set(COMMANDS) - _registry_commands()
    assert not missing, f"commands absent from reference/tool_registry.md: {sorted(missing)}"


def test_registry_lists_no_unknown_commands():
    unknown = _registry_commands() - set(COMMANDS)
    assert not unknown, f"tool_registry.md references unknown cta commands: {sorted(unknown)}"


def test_every_command_module_exists():
    pkg_dir = pathlib.Path(cta.__file__).resolve().parent
    for cmd, module in COMMANDS.items():
        stem = module.split(".")[-1]
        assert (pkg_dir / f"{stem}.py").exists(), f"{cmd} -> missing module {module}"
