# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest
from pathlib import Path


class HeadersTester(unittest.TestCase):

    def setUp(self):
        self.header_shebang = "#!usr/bin/python\n"
        self.header_coding = "# -*- coding: utf-8 -*-\n"
        self.header_blank = "\n"

        self.header_notice = ("# Copyright (c) Pyronear contributors.\n"
                              "# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.\n"
                              "# See the LICENSE file in the root of this repository for complete details.\n")

        self.excluded_files = ["version.py"]

    def test_first_two_lines_are_valid(self):
        # For every python file in the repository
        for source_path in Path(__file__).parent.parent.rglob('*.py'):
            if source_path.name not in self.excluded_files:
                with open(source_path) as source_file:
                    try:
                        first_two_lines = (next(source_file), next(source_file))
                    except StopIteration:
                        raise ValueError(f"Less than two lines in {source_path}. cannot check for shebang/encoding")

                    error_msg = f"\nInvalid first two lines in {source_path}"
                    self.assertIn(first_two_lines, [(self.header_shebang, self.header_coding),
                                                    (self.header_coding, self.header_blank)], msg=error_msg)

    def test_license_headers(self):
        """Test if license headers are correctly added at beginning of each file"""
        for source_path in Path(__file__).parent.parent.rglob('*.py'):
            if source_path.name not in self.excluded_files:
                with open(source_path) as source_file:
                    first_six_lines = ''.join(next(source_file) for _ in range(6))

                    error_msg = (f"\nHeader notice:\n{self.header_notice}\n"
                                 f"not found in first six lines of {source_path}:\n{first_six_lines}\n")
                    self.assertTrue(self.header_notice in first_six_lines, msg=error_msg)


if __name__ == '__main__':
    unittest.main()
