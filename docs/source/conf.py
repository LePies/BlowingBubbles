# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import shutil
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BlowingBubbles"
copyright = "2026, OliverMohr"
author = "OliverMohr"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# "myst_nb" registers the MyST parser itself (listing "myst_parser" as well
# would run its setup twice and fail), so only "myst_nb" is listed here.
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
# myst_nb registers ".ipynb" as a source suffix; keep the sphinx-gallery
# generated files in auto_examples/ (except the .rst pages) from being read
# as duplicate sources.
exclude_patterns = [
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
    "auto_examples/*.md5",
    "auto_examples/*.codeobj.json",
]

napoleon_use_ivar = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_logo = "_static/icon.svg"
html_favicon = "_static/icon.png"

# -- Auto-api setup ----------------------------------------------------------

autoapi_dirs = ["../../src/blowing_bubbles"]
# index.rst lists subpackages next to blowing_bubbles
autoapi_template_dir = "_templates/autoapi"
autoapi_ignore = ["*/examples/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]  # default minus 'imported-members'


# -- Demo notebooks (MyST-NB) -------------------------------------------------
# The demo notebooks live in examples/notebooks/ next to the sphinx-gallery
# scripts and are copied into the source tree at build time (Sphinx can only
# read sources inside the source directory). The copy is git-ignored; the
# notebooks are executed manually (jupyter) and rendered from their stored
# outputs, so the docs build does not need a kernel.

_source_dir = Path(__file__).resolve().parent
_examples_notebooks = _source_dir.parent.parent / "examples" / "notebooks"
_notebooks_dir = _source_dir / "notebooks"
if _examples_notebooks.is_dir():
    shutil.copytree(_examples_notebooks, _notebooks_dir, dirs_exist_ok=True)

nb_execution_mode = "off"


# -- Sphinx-Gallery ----------------------------------------------------------
# runs ../../examples/ scripts; output under docs/source/auto_examples/

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": "auto_examples",
    # Match example basenames on Windows and Unix (not only paths with "/").
    "filename_pattern": r"\.py$",
    "within_subsection_order": "FileNameSortKey",
    "capture_repr": (),
}


# -- BibTeX ------------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"


# -- Project to-do list (todo.rst) -------------------------------------------

project_root = Path(__file__).resolve().parent.parent.parent
TODO_MD_PATH = project_root / "todo.md"


def _parse_todo_md(lines):
    """Parse markdown checklist into (depth, checked, text) with $...$ -> :math:."""
    items = []
    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue
        m = re.match(r"^(\s*)- \[([ xX])\]\s*(.*)$", line)
        if m:
            indent, checked, rest = m.groups()
            depth = len(indent) // 4
            rest = re.sub(r"\$([^$]+)\$", r":math:`\1`", rest)
            items.append((depth, checked.lower() == "x", rest))
    return items


def _make_paragraph_with_inline(state, text, source):
    content = StringList([text, ""], source=source)
    wrapper = nodes.container()
    state.nested_parse(content, 0, wrapper)
    if len(wrapper.children) > 0:
        return wrapper[0]
    return nodes.paragraph("", nodes.Text(text))


class TodoListDirective(Directive):
    """Include the project to-do list from a markdown file, read at build time."""

    has_content = False
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        path = TODO_MD_PATH
        if self.arguments:
            path = project_root / self.arguments[0].strip()
        if not path.exists():
            return [nodes.warning("", nodes.Text(f"Todo file not found: {path}"))]
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("#"):
            lines = lines[1:]
        items = _parse_todo_md(lines)
        if not items:
            return [nodes.container()]

        source = str(path)
        wrapper = nodes.container()
        root_ul = nodes.bullet_list()
        wrapper += root_ul
        stack = [(root_ul, -1)]

        for depth, checked, rest in items:
            symbol = "✓ " if checked else "☐ "
            para = _make_paragraph_with_inline(self.state, symbol + rest, source)
            li = nodes.list_item("", para)
            while len(stack) > 1 and depth <= stack[-1][1]:
                stack.pop()
            current_ul, current_depth = stack[-1]
            if depth > 0 and depth > current_depth and len(current_ul.children) > 0:
                last_item = current_ul[-1]
                nested_ul = nodes.bullet_list()
                last_item += nested_ul
                nested_ul += li
                stack.append((nested_ul, depth))
            else:
                current_ul += li

        return [wrapper]


def setup(app):
    app.add_directive("todo-list", TodoListDirective)
