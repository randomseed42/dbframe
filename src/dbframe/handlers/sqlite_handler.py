import pandas as pd
import sqlalchemy as sa


class SQLiteDFHandler:
    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    def select(self):
        print(f'sqlite {self.conn_str} select')

    def insert(self):
        print(f'sqlite {self.conn_str} insert')
