#!/usr/bin/env python3
"""
yaml_io.py -- explicit YAML loading.

v7.2 problem
------------
`load_yaml_simple()` swallowed `ImportError` and returned `{}`. When PyYAML was
not installed the artist deconstruction table and the auto-exclusion table both
went silently empty, the pipeline still produced a prompt, and the user had no
idea half the configuration was missing.

v7.3 contract
-------------
Missing PyYAML is an explicit, reported condition. Callers choose what to do:

    * the compiler treats it as fatal (it cannot work without the tables)
    * the analyzer records the reason and emits no theory tags

Either way it is never silent.
"""


class YamlUnavailable(RuntimeError):
    """Raised when PyYAML is not installed."""


class YamlInvalid(RuntimeError):
    """Raised when a config file exists but cannot be parsed."""


INSTALL_HINT = "pip install pyyaml"


def pyyaml_available():
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        return False


def load_yaml(path, required=True):
    """
    Load a YAML file.

    Returns {} only when the file genuinely does not exist and `required` is
    False. A missing parser raises YamlUnavailable; a malformed file raises
    YamlInvalid. Nothing is swallowed.
    """
    import os

    if not path or not os.path.exists(path):
        if required:
            raise YamlInvalid(f"Config file not found: {path}")
        return {}

    try:
        import yaml
    except Exception as exc:
        raise YamlUnavailable(
            f"PyYAML is required to read {os.path.basename(path)}. Install it with: "
            f"{INSTALL_HINT}  (original error: {exc})"
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        raise YamlInvalid(f"Could not parse {path}: {exc}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise YamlInvalid(f"{path} must contain a YAML mapping at the top level.")
    return data
