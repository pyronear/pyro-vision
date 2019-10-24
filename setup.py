#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Package installation setup
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages


version = '0.1.0rc2'
sha = 'Unknown'
package_name = 'pyronear'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'pyronear', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name=package_name,
    version=version,
    author='PyroNear Contributors',
    author_email='pyronear.d4g@gmail.com',
    maintainer='François-Guillaume Fernandez',
    description='Datasets and models for wildfire detection in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/PyroNear',
    download_url='https://github.com/frgfm/PyroNear/tags',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['pytorch', 'deep learning', 'vision', 'models',
              'wildfire', 'object detection'],

    # Package info
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=requirements,
    package_data={'': ['LICENSE']}
)
