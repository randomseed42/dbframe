import os
import unittest

from dbframe.logger import Logger


class TestLogger(unittest.TestCase):
    def test_logger_1(self):
        logger = Logger().get_logger()
        with self.assertLogs(logger, level='WARNING') as log_capture:
            logger.debug('this is a debug log')
            logger.info('this is a info log')
            logger.warning('this is a warning log')
            logger.error('this is a error log')
            logger.critical('this is a critical log')
            self.assertIn('WARNING:Logger:this is a warning log', log_capture.output)
            self.assertIn('ERROR:Logger:this is a error log', log_capture.output)
            self.assertIn('CRITICAL:Logger:this is a critical log', log_capture.output)

    def test_logger_2(self):
        logger_conf = dict(
            log_name='Logger_Test',
            log_level=None,
            log_console=False,
            log_file='logger_test.log',
        )
        logger = Logger(**logger_conf).get_logger()
        logger.debug('this is a debug log')
        logger.info('this is a info log')
        logger.warning('this is a warning log')
        logger.error('this is a error log')
        logger.critical('this is a critical log')
        self.assertTrue(os.path.exists(logger_conf.get('log_file')))
        for hdlr in logger.handlers:
            hdlr.close()
        os.remove(logger_conf.get('log_file'))

    def test_logger_3(self):
        logger_conf = dict(
            log_name='Logger_Test2',
            log_level='debug',
            log_console=True,
            log_file=None,
        )
        logger = Logger(**logger_conf).get_logger()
        with self.assertLogs(logger, level='DEBUG') as log_capture:
            logger.debug('this is a debug log')
            logger.info('this is a info log')
            logger.warning('this is a warning log')
            logger.error('this is a error log')
            logger.critical('this is a critical log')
            self.assertIn('DEBUG:Logger_Test2:this is a debug log', log_capture.output)
            self.assertIn('INFO:Logger_Test2:this is a info log', log_capture.output)
            self.assertIn('WARNING:Logger_Test2:this is a warning log', log_capture.output)
            self.assertIn('ERROR:Logger_Test2:this is a error log', log_capture.output)
            self.assertIn('CRITICAL:Logger_Test2:this is a critical log', log_capture.output)
