#!usr/bin/python

# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

"""
Package installation setup
"""

import os
import subprocess
from setuptools import setup, find_packages


package_name = 'pyrovision'
with open(os.path.join('pyrovision', 'version.py')) as version_file:
    version = version_file.read().strip()
sha = 'Unknown'

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


with open('README.md') as f:
    readme = f.read()

requirements = [
    'opencv-python>=3.4.5.20',
    'pandas>=0.25.2',
    'torch>=1.8.0',
    'torchvision>=0.9.0',
    'tqdm>=4.20.0',
    'requests>=2.20.0',
    'pylocron>=0.1.3',
]

setup(
    # Metadata
    name=package_name,
    version=version,
    author='PyroNear Contributors',
    author_email='pyronear.d4g@gmail.com',
    maintainer='Pyronear',
    description='Datasets and models for wildfire detection in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/pyronear/pyro-vision',
    download_url='https://github.com/pyronear/pyro-vision/tags',
    license='AGPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: GNU Affero General Public License v3",
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
