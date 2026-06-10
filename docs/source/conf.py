# Configuration file for the Sphinx documentation builder.

from datetime import datetime

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------

project = "LLM4Series"
copyright = f"2026-{datetime.now().year} (MIT License)"
author = "Wesley Barbosa"
release = "1.0.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "numpydoc",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

language = "en"

autosummary_generate = True

add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "member-order": "bysource",
}

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "attrs_inline",
]

suppress_warnings = [
    "autosectionlabel.*",
]

# -----------------------------------------------------------------------------
# Copy Button
# -----------------------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -----------------------------------------------------------------------------
# Jupyter Notebooks
# -----------------------------------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_timeout = 600

# -----------------------------------------------------------------------------
# Intersphinx
# -----------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -----------------------------------------------------------------------------
# HTML Theme
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_logo = "_static/LLM4Series.svg"
html_favicon = "_static/logo.svg"

html_theme_options = {
    "show_prev_next": False,

    "navbar_start": [
        "navbar-logo",
    ],

    "navbar_center": [
        "navbar-nav",
    ],

    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],

    "show_toc_level": 2,

    "secondary_sidebar_items": [
        "page-toc",
    ],

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/llm4time/llm4series",
            "icon": "fab fa-github",
        },
    ],

    "pygments_light_style": "vs",
    "pygments_dark_style": "native",
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_show_sourcelink = False
