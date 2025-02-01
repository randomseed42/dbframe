import pandas as pd
import psycopg2
import sqlalchemy as sa


class PGDFHandler:
    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    def select(self):
        print(f'postgres {self.conn_str} select')

    def insert(self):
        print(f'postgres {self.conn_str} insert')
