from typing import Literal

from .handlers.postgres_handler import PGDFHandler
from .handlers.sqlite_handler import SQLiteDFHandler


class DBHandler:
    def __init__(self, db_type: Literal['sqlite', 'postgres'], conn_str: str):
        if db_type == 'sqlite':
            self.db = SQLiteDFHandler(conn_str=conn_str)
        elif db_type == 'postgres':
            self.db = PGDFHandler(conn_str=conn_str)
        else:
            raise ValueError(f'Unsupported database type: {db_type}')

    def select(self):
        self.db.select()

    def insert(self):
        self.db.insert()
