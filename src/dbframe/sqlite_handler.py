import os

import pandas as pd
from sqlalchemy import create_engine


class SQLiteHandler:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.url = self._generate_url()
        self.engine = create_engine(self.url, isolation_level='AUTOCOMMIT')

    def _generate_url(self):
        url_template = 'sqlite:///{}'
        if self.db_path is None or self.db_path == ":memory:":
            return url_template.format(':memory:')
        return url_template.format(os.path.abspath(self.db_path))

    def select(self):
        print(f'sqlite {self.url} select')

    def insert(self):
        print(f'sqlite {self.url} insert')


class SQLiteDFHandler(SQLiteHandler):
    def __init__(self, db_path: str = None):
        super().__init__(db_path=db_path)
