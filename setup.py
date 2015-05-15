#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.coding import setup
    setup

# Use UTF-8 if Python 3.
major, minor1, minor2, release, serial = sys.version_info
def read(filename):
    kwargs = {'encoding': 'utf-8'} if major > 3 else {}
    with open(filename, **kwargs):
        return f.read()

name = 'ptemcee'

# Get current version.
pattern = re.compile('__version__\s*=\s*(\'|")(.*?)(\'|")')
initPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptemcee/__init__.py')
pattern.findall(read(initPath))[0][1]

setup(
    name=name,
    version=version,
    author='Will Farr',
    author_email='w.farr@bham.ac.uk',
    packages=['ptemcee'],
    url='???',
    license='???',
    description='???',
    long_description='???',
    package_data={'': ['LICENSE', 'AUTHORS.rst']},
    include_package_data=True,
    install_requires=['numpy']
)
