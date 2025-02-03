import os
import unittest

import dbframe


class TestDBFrame(unittest.TestCase):
    def test_dbframe_file_path(self):
        pythonpath = os.getenv('PYTHONPATH', None)
        if pythonpath is None:
            self.assertTrue('site-packages' in dbframe.__file__)
        elif pythonpath == 'src':
            self.assertTrue('src' in dbframe.__file__)
        else:
            raise ImportError(f'Please check package source: {dbframe.__file__}')


if __name__ == '__main__':
    unittest.main()
