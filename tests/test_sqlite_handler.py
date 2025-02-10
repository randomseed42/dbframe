import os
import unittest
from datetime import datetime
from logging import FileHandler

from src.dbframe import SQLiteDFHandler
from src.dbframe.utils import WhereClause, OrderByClause
from sqlalchemy import Column, INTEGER, TEXT, TIMESTAMP


class TestSQLiteHandler(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='SQLite_Logger',
        log_level='CRITICAL',
        log_console=True,
        log_file='sqlite.log',
    )

    @staticmethod
    def _generate_columns() -> list[Column]:
        return [
            Column('uid', INTEGER()),
            Column('user', TEXT()),
            Column('register_datetime', TIMESTAMP()),
        ]

    @classmethod
    def _create_table(cls):
        cls.table_name = 'users'
        cls.columns = cls._generate_columns()
        rows = [
            dict(uid=0, user='user1', register_datetime=datetime(2020, 1, 1, 0, 0, 0)),
            dict(uid=1, user='user2', register_datetime=datetime(2020, 1, 2, 0, 0, 0)),
            dict(uid=2, user='user3', register_datetime=datetime(2020, 1, 3, 0, 0, 0)),
        ]
        cls.db.create_table(table_name=cls.table_name, columns=cls.columns)
        cls.db.insert_rows(table_name=cls.table_name, rows=rows, on_conflict='do_nothing')

    @classmethod
    def setUpClass(cls):
        cls.db_path = './test.db'
        cls.db = SQLiteDFHandler(db_path=cls.db_path, **cls.LOGGER_CONF)
        cls.table_name = 'users'
        cls._create_table()

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()
        os.remove(cls.db_path)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def test_db_init(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_db_get_tables(self):
        tables = self.db.get_tables()
        self.assertTrue(self.table_name in tables)

    def test_db_get_table(self):
        table = self.db.get_table(table_name=self.table_name)
        self.assertTrue(table.name == self.table_name)

    def test_db_get_column(self):
        column = self.db.get_column(table_name=self.table_name, column_name='user')
        self.assertEqual(column.name, 'user')
        column2 = self.db.get_column(table_name=self.table_name, column_name='user2')
        self.assertEqual(column2, None)

    def test_db_get_columns(self):
        columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue(set(col.name for col in self.columns).issubset(columns.keys()))

    def test_db_create_rename_drop_table(self):
        temp_table = 'temp_table'
        table = self.db.create_table(table_name=temp_table, columns=self._generate_columns())
        self.assertEqual(table.name, temp_table)

        new_temp_table = 'new_temp_table'
        self.db.rename_table(old_table_name=temp_table, new_table_name=new_temp_table)
        self.assertTrue(self.db.get_table(table_name=temp_table) is None)
        self.assertTrue(self.db.get_table(table_name=new_temp_table) is not None)

        self.db.drop_table(table_name=new_temp_table)
        self.assertIsNone(self.db.get_table(table_name=new_temp_table))

        with self.assertRaisesRegex(ValueError, 'Table.*already exists.*'):
            self.db.create_table(table_name='users', columns=self._generate_columns())

    def test_db_add_column(self):
        new_column = Column('age', INTEGER())
        self.db.add_column(table_name=self.table_name, column=new_column)
        self.assertTrue('age' in self.db.get_columns(table_name=self.table_name))

    def test_db_add_columns(self):
        new_columns = [
            Column('uid', INTEGER()),
            Column('first_name', TEXT()),
            Column('last_name', TEXT()),
        ]
        self.db.add_columns(table_name=self.table_name, columns=new_columns)
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('first_name' in current_columns and 'last_name' in current_columns)

    def test_db_drop_column(self):
        new_column = Column('email', TEXT())
        self.db.add_column(table_name=self.table_name, column=new_column)
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('email' in current_columns)
        self.db.drop_column(table_name=self.table_name, column_name='email')
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('email' not in current_columns)

    def test_db_drop_columns(self):
        new_columns = [
            Column('password', TEXT()),
            Column('role', TEXT()),
        ]
        self.db.add_columns(table_name=self.table_name, columns=new_columns)
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('password' in current_columns and 'role' in current_columns)
        self.db.drop_columns(table_name=self.table_name, columns=['password', 'role'])
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('password' not in current_columns and 'role' not in current_columns)

    # def test_db_alter_column(self):
    #     new_columns = [
    #         Column('sex', INTEGER()),
    #         Column('first_name', TEXT()),
    #         Column('last_name', TEXT()),
    #     ]
    #     self.db.add_columns(table_name=self.table_name, columns=new_columns)
    #     self.db.alter_column(table_name=self.table_name, column='sex', new_column_name='gender')
    #     self.db.alter_column(table_name=self.table_name, column='last_name', new_column_name='family_name')
    #     current_columns = self.db.get_columns(table_name=self.table_name)
    #     self.assertTrue('gender' in current_columns and 'sex' not in current_columns)
    #     self.assertTrue('family_name' in current_columns and 'last_name' not in current_columns)

    def test_db_get_indexes(self):
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertEqual(len(indexes), 0)
        self.db.create_index(table_name=self.table_name, columns=['uid', 'user'])
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertTrue(f'ix_{self.table_name}_uid_user' in indexes)
        with self.assertRaisesRegex(ValueError, 'Index .* already exists'):
            self.db.create_index(table_name=self.table_name, columns=['uid', 'user'])
        with self.assertRaisesRegex(ValueError, 'Column .* not in'):
            self.db.create_index(table_name=self.table_name, columns=['user', 'email'])

    def test_db_create_index(self):
        self.db.create_index(table_name=self.table_name, columns=['user', 'register_datetime'])
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertTrue(f'ix_{self.table_name}_user_register_datetime' in indexes)
        self.db.drop_index(table_name=self.table_name, columns=['user', 'register_datetime'])

    def test_db_drop_index(self):
        self.db.create_index(table_name=self.table_name, columns=['register_datetime'])
        self.db.drop_index(table_name=self.table_name, columns=['register_datetime'])
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertTrue(f'ix_{self.table_name}_register_datetime' not in indexes)

    def test_db_insert_rows(self):
        rows = [
            dict(uid=3, user='user4', register_datetime=datetime(2020, 1, 4, 0, 0, 0)),
            dict(uid=4, user='user5', register_datetime=datetime(2020, 1, 5, 0, 0, 0)),
        ]
        self.db.insert_rows(table_name=self.table_name, rows=rows, on_conflict='do_nothing')
        cols, rows = self.db.select_rows(table_name=self.table_name, columns=['uid', 'user'])
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 5)

    # def test_db_select_rows(self):
    #     cols, rows = self.db.select_rows(table_name=self.table_name)
    #     self.assertEqual(cols, ['uid', 'user', 'register_datetime', 'age', 'first_name', 'family_name', 'gender'])
    #     self.assertEqual(len(rows), 5)
    #
    #     cols, rows = self.db.select_rows(table_name=self.table_name, columns=['uid', 'user', 'email'])
    #     self.assertEqual(cols, ['uid', 'user'])
    #     self.assertEqual(rows, [(0, 'user1'), (1, 'user2'), (2, 'user3'), (3, 'user4'), (4, 'user5')])
    #
    #     cols, rows = self.db.select_rows(table_name=self.table_name, columns=['email'])
    #     self.assertEqual(cols, None)
    #     self.assertEqual(rows, None)

    def test_db_select_rows_where(self):
        where_clauses = WhereClause('uid', '>', 2)
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 2)

        where_clauses = [WhereClause('uid', '>=', 2)]
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

        where_clauses = ([WhereClause('uid', '>', 0), WhereClause('uid', '<', 2)], WhereClause('uid', '>', 3))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 2)

        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 4)

        where_clauses = (WhereClause('user', 'like', 'User1'), WhereClause('user', 'ilike', 'User2'), WhereClause('user', 'ilike', 'user3'), WhereClause('user', 'ilike', '%SER4'))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 4)

    def test_db_select_rows_where_order(self):
        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        order_by = [OrderByClause('uid', False)]
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses,
                                         order_by=order_by)
        self.assertEqual(len(rows), 4)

    def test_db_select_rows_where_order_offset_limit(self):
        where_clauses = None
        order_by = [OrderByClause('uid', False)]
        offset = 1
        limit = 3
        cols, rows = self.db.select_rows(
            self.table_name,
            ['uid', 'user', 'email'],
            where_clauses=where_clauses,
            order_by=order_by,
            offset=offset,
            limit=limit,
        )
        self.assertEqual(len(rows), 3)

    def test_db_delete_rows(self):
        rows = [
            dict(uid=5, user='user6', register_datetime=datetime(2020, 1, 6, 0, 0, 0)),
            dict(uid=6, user='user7', register_datetime=datetime(2020, 1, 7, 0, 0, 0)),
        ]
        where_clauses = (WhereClause('uid', '==', 5), WhereClause('user', '==', 'user7'))
        self.db.insert_rows(table_name=self.table_name, rows=rows, on_conflict='do_nothing')
        self.db.delete_rows(table_name=self.table_name, where_clauses=where_clauses)
        cols, rows = self.db.select_rows(table_name=self.table_name, where_clauses=where_clauses)
        self.assertEqual(rows, [])

    def test_db_update_rows(self):
        where_clauses = (WhereClause('uid', '==', 2), WhereClause('user', '==', 'user4'))
        self.db.update_rows(table_name=self.table_name, set_clauses={'register_datetime': datetime(2020, 2, 1)},
                            where_clauses=where_clauses)
        cols, rows = self.db.select_rows(table_name=self.table_name, columns=['register_datetime'],
                                         where_clauses=where_clauses)
        self.assertEqual(set(rows), {(datetime(2020, 2, 1),)})

    def test_db_execute_sql(self):
        sql = 'SELECT uid, user AS name FROM users'
        cols, rows = self.db._execute_sql(sql, return_result=True)
        self.assertEqual(cols, ['uid', 'name'])


if __name__ == '__main__':
    unittest.main()
