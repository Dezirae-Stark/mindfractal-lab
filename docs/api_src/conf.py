"""
Sphinx Configuration for MindFractal Lab API Documentation
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MindFractal Lab'
copyright = '2025, MindFractal Lab Contributors'
author = 'MindFractal Lab Contributors'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# AutoDoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master doc
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    'display_github': True,
    'github_user': 'Dezirae-Stark',
    'github_repo': 'mindfractal-lab',
    'github_version': 'main',
    'conf_py_path': '/docs/api_src/',
}

# -- MathJax configuration ---------------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            'R': r'\mathbb{R}',
            'C': r'\mathbb{C}',
            'vx': r'\mathbf{x}',
            'vz': r'\mathbf{z}',
            'vc': r'\mathbf{c}',
            'mA': r'\mathbf{A}',
            'mB': r'\mathbf{B}',
            'mW': r'\mathbf{W}',
        }
    }
}

# -- Extension configuration -------------------------------------------------

# MyST parser settings
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'colon_fence',
]
