import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Phentax"

import phentax

copyright = phentax.__copyright__
author = phentax.__author__
version = phentax.__version__

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",  # Creates .nojekyll file
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_custom_sections = ["Method"]

typehints_defaults = "comma"
always_use_bars_union = True

mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {
        "packages": {"[+]": ["physics"]},
        "macros": {},
    },
}

autosummary_generate = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
