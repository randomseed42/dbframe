import os
import unittest
from datetime import datetime
from logging import FileHandler

from dbframe import SQLiteDFHandler
from sqlalchemy import Column, INTEGER, TEXT, DATETIME


class TestSQLiteHandler(unittest.TestCase):
    @staticmethod
    def _generate_columns():
        columns = [
            Column('uid', INTEGER()),
            Column('user', TEXT()),
            Column('register_datetime', DATETIME()),
        ]
        return columns

    @classmethod
    def setUpClass(cls):
        cls.db_path = './test.db'
        cls.logger_conf = dict(
            log_name='Logger',
            log_level='DEBUG',
            log_console=True,
            log_file='db.log',
        )
        cls.db = SQLiteDFHandler(db_path=cls.db_path, **cls.logger_conf)
        cls.table_name = 'users'
        cls.columns = cls._generate_columns()
        cls.db.create_table(table_name=cls.table_name, columns=cls.columns)

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()
        os.remove(cls.db_path)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.logger_conf.get('log_file'))

    def test_db_init(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_db_get_tables(self):
        tables = self.db.get_tables()
        self.assertTrue(self.table_name in tables)

    def test_db_get_table(self):
        table = self.db.get_table(table_name=self.table_name)
        self.assertTrue(table.name == self.table_name)

    def test_db_get_columns(self):
        columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue(set(col.name for col in self.columns).issubset(columns.keys()))

    def test_db_create_drop_table(self):
        name = 'users2'
        table = self.db.create_table(table_name=name, columns=self._generate_columns())
        self.assertEqual(table.name, name)
        self.db.drop_table(table_name=name)
        self.assertIsNone(self.db.get_table(table_name=name))

    def test_db_insert_rows(self):
        rows = [
            dict(uid=0, user='user1', register_datetime=datetime(2020, 1, 1, 0, 0, 0)),
            dict(uid=1, user='user2', register_datetime=datetime(2020, 1, 2, 0, 0, 0)),
        ]
        self.db.insert_rows(table_name=self.table_name, rows=rows, on_conflict='do_nothing')

    def test_db_add_column(self):
        new_column = Column('age', INTEGER())
        self.db.add_column(table_name=self.table_name, column=new_column)
        self.assertTrue('age' in self.db.get_columns(table_name=self.table_name))


if __name__ == '__main__':
    unittest.main()
