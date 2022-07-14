# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os

import pyrovision

PROJECT_NAME: str = "PyroVision API template"
PROJECT_DESCRIPTION: str = "Template API for Computer Vision"
VERSION: str = pyrovision.__version__
DEBUG: bool = os.environ.get("DEBUG", "") != "False"
