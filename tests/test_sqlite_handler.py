import os
import unittest
from datetime import datetime

from dbframe import SQLiteDFHandler
from sqlalchemy import Column, Integer, String, DateTime


class TestSQLiteHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = './test.db'
        cls.db = SQLiteDFHandler(db_path=cls.db_path)
        cls.db.create_table(table_name='users', columns=[
            Column('id', Integer),
            Column('user', String),
            Column('register_datetime', DateTime),
        ])

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()
        os.remove(cls.db_path)

    def test_db_init(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_db_get_tables(self):
        tables = self.db.get_tables()
        self.assertTrue('users' in tables)

    def test_db_get_table(self):
        table = self.db.get_table(table_name='users')
        self.assertTrue(table.name == 'users')

    def test_db_get_columns(self):
        columns = self.db.get_columns(table_name='users')
        self.assertEqual(list(columns.keys()), ['id', 'user', 'register_datetime'])

    def test_db_create_table(self):
        table = self.db.create_table(table_name='users', columns=[])
        self.assertEqual(table.name, 'users')

    def test_db_insert_rows(self):
        rows = [
            dict(id=0, user='user1', register_datetime=datetime(2020, 1, 1, 0, 0, 0)),
            dict(id=1, user='user2', register_datetime=datetime(2020, 1, 2, 0, 0, 0)),
        ]
        self.db.insert_rows(table_name='users', rows=rows, on_conflict='do_nothing')

if __name__ == '__main__':
    unittest.main()
