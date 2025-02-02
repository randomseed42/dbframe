import os

from sqlalchemy import create_engine

from .logger import Logger

logger = Logger('SQLite_Logger').get_logger()


class SQLiteHandler:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.url = self._generate_url()
        self.engine = create_engine(self.url, isolation_level='AUTOCOMMIT')
        self._validate_connection()

    def _generate_url(self):
        url_template = 'sqlite:///{}'
        if self.db_path is None or self.db_path == ":memory:":
            return url_template.format(':memory:')
        return url_template.format(os.path.abspath(self.db_path))

    def _validate_connection(self):
        try:
            self.engine.connect()
            logger.info('Connection successful')
        except Exception as err:
            logger.error(str(err))
            raise ValueError(f'Connection failed: {self.db_path=}')

    def select(self):
        print(f'sqlite {self.url} select')

    def insert(self):
        print(f'sqlite {self.url} insert')


class SQLiteDFHandler(SQLiteHandler):
    def __init__(self, db_path: str = None):
        super().__init__(db_path=db_path)
