from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, file_predicate
from sphinx_polyversion.pyvenv import Poetry
from sphinx_polyversion.sphinx import SphinxBuilder

#: Regex matching the branches to build docs for
BRANCH_REGEX = r"(multi_sphinx)"

#: Regex matching the tags to build docs for
TAG_REGEX = r""

#: Output dir relative to project root
OUTPUT_DIR = "docs/build"

#: Source directory
SOURCE_DIR = "docs"

#: Arguments to pass to `poetry install`
POETRY_ARGS = "--only sphinx --sync".split()

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = []

#: Mock data used for building local version
MOCK_DATA = {}
#: Whether to build using only local files and mock data
MOCK = False

#: Whether to run the builds in sequence or in parallel
SEQUENTIAL = False

# Load overrides read from commandline to global scope
apply_overrides(globals())
# Determine repository root directory
root = Git.root(Path(__file__).parent)

# Setup driver and run it
src = Path(SOURCE_DIR)
DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        buffer_size=1 * 10**9,  # 1 GB
        predicate=file_predicate([src]), # exclude refs without source dir
    ),
    builder=SphinxBuilder(src / "sphinx", args=SPHINX_ARGS),
    env=Poetry.factory(args=POETRY_ARGS),
    template_dir=root / src / "templates",
    static_dir=root / src / "static",
    mock=MOCK_DATA,
).run(MOCK)
