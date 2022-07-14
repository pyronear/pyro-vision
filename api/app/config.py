# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os

PROJECT_NAME: str = "PyroVision API template"
PROJECT_DESCRIPTION: str = "Template API for Computer Vision"
VERSION: str = "0.2.0.dev0"
DEBUG: bool = os.environ.get("DEBUG", "") != "False"
HUB_REPO: str = "pyronear/rexnet1_0x"
