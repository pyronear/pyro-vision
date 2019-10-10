import unittest
from pyronear.utils.collect_env import get_pretty_env_info


class TestCollectEnv(unittest.TestCase):
    def test_prettyenv(self):
        info_output = get_pretty_env_info()
        self.assertTrue(info_output.count('\n') >= 19)


if __name__ == '__main__':
    unittest.main()