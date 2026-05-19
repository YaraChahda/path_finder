"""Sphinx configuration."""
import os
import shutil
import sys
from importlib.metadata import metadata

# -- Path setup ---------------------------------------------------------------

__location__ = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(__location__, "../../src"))

# -- Run sphinx-apidoc --------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See:
# https://github.com/readthedocs/readthedocs.org/issues/1139

try:
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "../../src/path_finder")

try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    import sphinx
    cmd_line = f"sphinx-apidoc --implicit-namespaces -f -o {output_dir} {module_dir}"
    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        args = args[1:]
    apidoc.main(args)
except Exception as e:
    print(f"Running `sphinx-apidoc` failed!\n{e}")

# -- Project information ------------------------------------------------------

_metadata = metadata("path-finder-retrosynthesis")

project   = "path-finder-retrosynthesis"
author    = "Yara Chahda, Corentin Portmann, Ines Ouchen"
copyright = f"2024, {author}"
version   = _metadata["Version"]
release   = ".".join(version.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path   = ["_templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]
suppress_warnings = ["myst.header"]

# -- Options for HTML output --------------------------------------------------

html_theme       = "furo"
html_static_path = ["_static"]