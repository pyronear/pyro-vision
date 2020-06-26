# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest
from pyronear import utils


class UtilsTester(unittest.TestCase):
    def test_prettyenv(self):
        info_output = utils.get_pretty_env_info()
        self.assertTrue(info_output.count('\n') >= 19)


if __name__ == '__main__':
    unittest.main()
