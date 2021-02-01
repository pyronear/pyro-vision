# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import unittest
from pyrovision import utils


class UtilsTester(unittest.TestCase):
    def test_prettyenv(self):
        info_output = utils.get_pretty_env_info()
        self.assertTrue(info_output.count('\n') >= 19)


if __name__ == '__main__':
    unittest.main()
