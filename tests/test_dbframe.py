import os
import unittest
from logging import FileHandler

import dbframe
from dbframe.logger import Logger


class TestDBFrame(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='DBFrame_Logger',
        log_level='DEBUG',
        log_console=True,
        log_file='dbframe.log',
    )

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger(**cls.LOGGER_CONF).get_logger()

    @classmethod
    def tearDownClass(cls):
        for hdlr in cls.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def test_dbframe_file_path(self):
        pythonpath = os.getenv('PYTHONPATH', None)
        self.logger = Logger(**self.LOGGER_CONF).get_logger()
        if pythonpath is None:
            self.assertTrue('site-packages' in dbframe.__file__)
            self.logger.critical('This is post install test')
        elif pythonpath == 'src':
            self.assertTrue('src' in dbframe.__file__)
            self.logger.critical('This is development test')
        else:
            raise ImportError(f'Please check package source: {dbframe.__file__}')


if __name__ == '__main__':
    unittest.main()
