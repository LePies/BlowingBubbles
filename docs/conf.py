# BlowingBubbles — Sphinx configuration

import dataclasses
import inspect as py_inspect
import re
import sys
from enum import Enum
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.ext.autodoc import ClassDocumenter

project_root = Path(__file__).resolve().parent.parent
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
        try:
            env = self.state.document.settings.env
            srcdir = Path(env.srcdir)
            dep_path = path.resolve().relative_to(srcdir)
            env.note_dependency(str(dep_path.as_posix()))
        except (ValueError, AttributeError):
            pass
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("#"):
            lines = lines[1:]
        items = _parse_todo_md(lines)
        if not items:
            return [nodes.container(classes=["todo-list"])]

        source = str(path)
        wrapper = nodes.container(classes=["todo-list"])
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


project = "BlowingBubbles"
copyright = "2026, OliverMohr"
author = "OliverMohr"
release = "0.2.0"
version = "0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]

# Sphinx-Gallery: runs ../examples/ scripts; output under docs/auto_examples/
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": "auto_examples",
    # Match example basenames on Windows and Unix (not only paths with "/").
    "filename_pattern": r"\.py$",
    "within_subsection_order": "FileNameSortKey",
    "capture_repr": (),
}
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
add_module_names = False


def _detect_class_kind(obj):
    if not py_inspect.isclass(obj):
        return None
    if dataclasses.is_dataclass(obj):
        return "Data class"
    try:
        if issubclass(obj, Enum):
            return "Enum"
    except TypeError:
        pass
    return "Class"


class _LabeledClassDocumenter(ClassDocumenter):
    def add_directive_header(self, sig):
        if not self.doc_as_attr:
            kind = _detect_class_kind(self.object)
            if kind is not None:
                sourcename = self.get_sourcename()
                name = getattr(self.object, "__name__", None) or self.objpath[-1]
                self.add_line(f".. rubric:: {name} ({kind})", sourcename)
                self.add_line("   :class: api-class-header", sourcename)
                self.add_line("", sourcename)
        super().add_directive_header(sig)


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "BlowingBubbles"
html_short_title = "BlowingBubbles"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1a5fb4",
        "color-brand-content": "#1a5fb4",
        "font-stack": "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "font-size-normal": "16px",
        "sidebar-item-spacing-compact": "0.4rem",
    },
    "dark_css_variables": {
        "color-brand-primary": "#62a0ea",
        "color-brand-content": "#62a0ea",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
# html_logo not set; to add a package icon see "Package icon" in getting_started.rst
html_favicon = None
html_show_sourcelink = True
html_show_sphinx = False


def setup(app):
    app.add_css_file("custom.css")
    app.add_directive("todo-list", TodoListDirective)
    app.add_autodocumenter(_LabeledClassDocumenter, override=True)

    def _add_project_path(_app):
        sys.path.insert(0, str(project_root / "src"))

    app.connect("builder-inited", _add_project_path)
