import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(REPO_ROOT, "scripts")

for path in (SCRIPTS, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def tag_library(repo_root):
    """The real suno_tag_library.yaml, parsed."""
    from yaml_io import load_yaml
    return load_yaml(os.path.join(repo_root, "suno_tag_library.yaml"), required=True)


@pytest.fixture(scope="session")
def exclusion_map(repo_root):
    """The real auto_exclusion_map.yaml, parsed."""
    from yaml_io import load_yaml
    return load_yaml(os.path.join(repo_root, "auto_exclusion_map.yaml"), required=True)


@pytest.fixture(scope="session")
def artist_map(repo_root):
    from yaml_io import load_yaml
    return load_yaml(os.path.join(repo_root, "artist_feature_map.yaml"), required=False)
