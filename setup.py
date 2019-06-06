#!/usr/bin/env python3
from __future__ import division

from setuptools import setup
from glob import glob
import ast
import re
import os, importlib
from setuptools import setup

__author__ = "The Clemente Lab"
__copyright__ = "Copyright 2019, The Clemente Lab"
__credits__ = ["Kevin Bu, Steve Schmerler"]
__license__ = "GPL"
__maintainer__ = "Kevin Bu"
__email__ = "kbu314@gmail.com"

# https://github.com/mitsuhiko/flask/blob/master/setup.py
_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('iclust/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(name='iclust',
      version=version,
      description='Image clustering',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.7.3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='http://github.com/clemente-lab/TBD',
      author=__author__,
      author_email=__email__,
      license=__license__,
      packages=['iclust'],
      scripts=glob('scripts/*py'),
      install_requires=[
          'tensorflow',
          'keras',
          'Pillow',
          'scipy',
          'matplotlib',
          'seaborn'
      ],
      zip_safe=False)




