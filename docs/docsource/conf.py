# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#import sphinx_pdj_theme
import sphinx_rtd_theme
#print(os.path.join(os.path.abspath('.'), '..', 'scenewalk'))
#sys.path.insert(0, os.path.join(os.path.abspath('.'), '..', 'scenewalk'))
sys.path.insert(0, "/Users/lisa/Documents/SFB1294_B05/SceneWalk/SceneWalk_model")
sys.path.insert(0, os.path.abspath('../demo'))
sys.path.insert(0, os.path.abspath('/Users/lisa/Documents/SFB1294_B05/PyDREAM'))
# -- Project information -----------------------------------------------------

project = 'SceneWalk'
copyright = '2020, Lisa Schwetlick'
author = 'Lisa Schwetlick'


# -- General configuration ---------------------------------------------------
source_suffix = ['.rst']

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
        "sphinx_rtd_theme",
              'numpydoc',
              'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_pdj_theme'
html_theme = "sphinx_rtd_theme"
#htm_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
pygments_style = 'sphinx'
html_logo = 'sw_logo2.png'
html_theme_options = {
'logo_only': True,
        }
html_css_files = ['css/custom_style.css']

html_sidebars = {}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
