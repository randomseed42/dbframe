import logging
import os
from functools import lru_cache

from .types import LogLevelType


class Logger:
    _default_log_level = 'WARNING'

    @lru_cache(maxsize=None)
    def __new__(cls, log_name: str = 'Logger', log_level: LogLevelType = None, log_console: bool = True, log_file: str = None):
        instance = super().__new__(cls)
        instance._setup_logger(log_name=log_name, log_level=log_level, log_console=log_console, log_file=log_file)
        return instance

    def _setup_logger(self, log_name: str, log_level: LogLevelType = None, log_console: bool = True, log_file: str = None):
        if log_level is None:
            log_level = os.getenv('LOG_LEVEL', self._default_log_level).upper()
        else:
            log_level = log_level.upper()
        log_level = getattr(logging, log_level, logging.WARNING)

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            '{asctime} | {name} | {levelname:<8} | {filename}:{lineno} | {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        if log_console:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
