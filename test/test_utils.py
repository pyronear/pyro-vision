import unittest
from pyronear import utils


class UtilsTester(unittest.TestCase):
    def test_prettyenv(self):
        info_output = utils.get_pretty_env_info()
        self.assertTrue(info_output.count('\n') >= 19)


if __name__ == '__main__':
    unittest.main()
