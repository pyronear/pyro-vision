#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Package installation setup
"""

import os
import sys

from setuptools import setup, find_packages


if sys.argv[-1] == 'publish':
    os.system('python3 setup.py sdist upload')
    sys.exit()

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name='pyronear',
    version='0.1.0a0',
    author='François-Guillaume Fernandez',
    description='Modules, operations and models for wildfire detection in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/PyroNear',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='pytorch deep learning vision models',

    # Package info
    packages=['pyronear'],
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=requirements,
    package_data={'': ['LICENSE']}
)