import os
import unittest
from datetime import datetime
from logging import FileHandler

from dbframe import SQLiteDFHandler
from dbframe.utils import WhereClause, OrderByClause
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
            Column('uid', INTEGER(), primary_key=True),
            Column('user', TEXT()),
            Column('register_datetime', TIMESTAMP()),
        ]

    @classmethod
    def _create_table(cls):
        cls.columns = cls._generate_columns()
        rows = [
            dict(uid=0, user='user1', register_datetime=datetime(2025, 1, 1, 0, 0, 0)),
            dict(uid=1, user='user2', register_datetime=datetime(2025, 1, 2, 0, 0, 0)),
            dict(uid=2, user='user3', register_datetime=datetime(2025, 1, 3, 0, 0, 0)),
            dict(uid=3, user='user4', register_datetime=datetime(2025, 1, 4, 0, 0, 0)),
            dict(uid=4, user='user5', register_datetime=datetime(2025, 1, 5, 0, 0, 0)),
        ]
        cls.db.create_table(table_name=cls.table_name, columns=cls.columns)
        cls.db.insert_rows(table_name=cls.table_name, rows=rows, on_conflict='do_nothing')

    @classmethod
    def _drop_table(cls):
        cls.db.drop_table(table_name=cls.table_name)

    @classmethod
    def setUpClass(cls):
        cls.db_path = './test.db'
        cls.table_name = 'users'
        cls.db = SQLiteDFHandler(db_path=cls.db_path, **cls.LOGGER_CONF)

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()
        os.remove(cls.db_path)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def setUp(self):
        self._create_table()

    def tearDown(self):
        self._drop_table()

    def test_db_init(self):
        self.assertTrue(os.path.exists(self.db_path))

    # Database CRUD
    def test_db_create_database(self):
        temp_db_path = './temp.db'
        abs_temp_db_path = self.db.create_database(db_path=temp_db_path)
        self.assertEqual(os.path.basename(abs_temp_db_path), 'temp.db')
        none_temp_db_path = self.db.create_database(db_path=temp_db_path)
        self.assertEqual(none_temp_db_path, None)
        os.remove(abs_temp_db_path)

    def test_db_get_database(self):
        temp_db_path = './temp.db'
        none_temp_db_path = self.db.get_database(db_path=temp_db_path)
        self.assertEqual(none_temp_db_path, None)
        abs_temp_db_path = self.db.create_database(db_path=temp_db_path)
        abs_temp_db_path2 = self.db.get_database(db_path=temp_db_path)
        self.assertEqual(abs_temp_db_path, abs_temp_db_path2)
        os.remove(abs_temp_db_path)

    def test_db_drop_database(self):
        temp_db_path = './temp.db'
        abs_temp_db_path = self.db.create_database(db_path=temp_db_path)
        abs_temp_db_path2 = self.db.drop_database(db_path=temp_db_path)
        self.assertEqual(abs_temp_db_path, abs_temp_db_path2)

    # Table CRUD
    def test_db_create_table(self):
        temp_table = 'temp_table'
        table = self.db.create_table(table_name=temp_table, columns=self._generate_columns())
        self.assertEqual(table.name, temp_table)

        with self.assertRaisesRegex(ValueError, 'Table.*already exists.*'):
            self.db.create_table(table_name=self.table_name, columns=self._generate_columns())

        with self.assertRaisesRegex(ValueError, 'Columns.*cannot be empty'):
            temp_table2 = 'temp_table2'
            self.db.create_table(table_name=temp_table2, columns=[])

    def test_db_get_table(self):
        table = self.db.get_table(table_name=self.table_name)
        self.assertTrue(table.name == self.table_name)

    def test_db_get_tables(self):
        tables = self.db.get_tables()
        self.assertTrue(self.table_name in tables)

    def test_db_rename_table(self):
        new_table_name = 'user2'
        self.db.rename_table(old_table_name=self.table_name, new_table_name=new_table_name)
        self.assertTrue(self.db.get_table(table_name=self.table_name) is None)
        self.assertEqual(self.db.get_table(table_name=new_table_name).name, new_table_name)

    def test_db_drop_table(self):
        self.db.drop_table(table_name=self.table_name)
        self.assertIsNone(self.db.get_table(table_name=self.table_name))

    # Column CRUD
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

    def test_db_get_column(self):
        column = self.db.get_column(table_name=self.table_name, column_name='UseR')
        self.assertEqual(column.name, 'user')
        column2 = self.db.get_column(table_name=self.table_name, column_name='user2')
        self.assertEqual(column2, None)

    def test_db_get_columns(self):
        columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue(set(col.name for col in self.columns).issubset(columns.keys()))

    def test_db_alter_column(self):
        column_name = self.db.alter_column(table_name=self.table_name, old_column_name='user',
                                           new_column_name='username')
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue(column_name in current_columns and 'user' not in current_columns)

    def test_db_drop_column(self):
        self.db.drop_column(table_name=self.table_name, column_name='user')
        self.assertEqual(self.db.get_column(table_name=self.table_name, column_name='user'), None)

    def test_db_drop_columns(self):
        self.db.drop_columns(table_name=self.table_name, column_names=['user', 'register_datetime'])
        current_columns = self.db.get_columns(table_name=self.table_name)
        self.assertTrue('user' not in current_columns and 'register_datetime' not in current_columns)

    # Index CRUD
    def test_db_create_index(self):
        self.db.create_index(table_name=self.table_name, column_names=['uid', 'user'])
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertTrue(f'ix_{self.table_name}_uid_user' in indexes)
        with self.assertRaisesRegex(ValueError, 'Index .* already exists'):
            self.db.create_index(table_name=self.table_name, column_names=['uid', 'user'])
        with self.assertRaisesRegex(ValueError, 'Column .* not in'):
            self.db.create_index(table_name=self.table_name, column_names=['user', 'email'])

    def test_db_get_indexes(self):
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertEqual(len(indexes), 0)
        self.db.create_index(table_name=self.table_name, column_names=['uid', 'user'])
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertEqual(indexes, {f'ix_{self.table_name}_uid_user': ['uid', 'user']})

    def test_db_drop_index(self):
        self.db.create_index(table_name=self.table_name, column_names=['uid', 'user'])
        idx_name = self.db.drop_index(table_name=self.table_name, column_names=['uid', 'user'])
        self.assertEqual(f'ix_{self.table_name}_uid_user', idx_name)
        indexes = self.db.get_indexes(table_name=self.table_name)
        self.assertTrue(idx_name not in indexes)

    # Rows CRUD
    def test_db_insert_rows(self):
        rows = [
            dict(uid=5, user='user6', register_datetime=datetime(2025, 1, 6, 0, 0, 0)),
            dict(uid=6, user='user7', register_datetime=datetime(2025, 1, 7, 0, 0, 0)),
        ]
        rowcount = self.db.insert_rows(table_name=self.table_name, rows=rows, on_conflict='do_nothing')
        self.assertEqual(rowcount, len(rows))

    def test_db_select_rows(self):
        cols, rows = self.db.select_rows(table_name=self.table_name)
        self.assertEqual(cols, ['uid', 'user', 'register_datetime'])
        self.assertEqual(len(rows), 5)

        cols, rows = self.db.select_rows(table_name=self.table_name, column_names=['uid', 'user', 'email'])
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(rows, [(0, 'user1'), (1, 'user2'), (2, 'user3'), (3, 'user4'), (4, 'user5')])

        cols, rows = self.db.select_rows(table_name=self.table_name, column_names=['email'])
        self.assertEqual((cols, rows), (None, None))

    def test_db_select_rows_where(self):
        where_clauses = WhereClause('uid', '<', 2)
        cols, rows = self.db.select_rows(table_name=self.table_name, column_names=['uid', 'user', 'email'],
                                         where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 2)

        where_clauses = [WhereClause('uid', '>=', 2)]
        cols, rows = self.db.select_rows(table_name=self.table_name, column_names=['uid', 'user', 'email'],
                                         where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

        where_clauses = ([WhereClause('uid', '>', 0), WhereClause('uid', '<', 2)], WhereClause('uid', '>', 2))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 4)

        where_clauses = (WhereClause('User', 'like', 'User1'), WhereClause('USER', 'ilike', 'User2'),
                         WhereClause('user', 'ilike', 'user3'), WhereClause('user', 'ilike', '%SER4'))
        cols, rows = self.db.select_rows(self.table_name, ['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 4)

    def test_db_select_rows_where_order(self):
        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        order_by = [OrderByClause('uid', False)]
        cols, rows = self.db.select_rows(table_name=self.table_name, column_names=['uid', 'user', 'email'],
                                         where_clauses=where_clauses, order_by=order_by)
        self.assertEqual(rows, [(4, 'user5'), (2, 'user3'), (1, 'user2'), (0, 'user1')])

    def test_db_select_rows_where_order_offset_limit(self):
        where_clauses = None
        order_by = [OrderByClause('uid', False)]
        offset = 1
        limit = 3
        cols, rows = self.db.select_rows(
            table_name=self.table_name,
            column_names=['uid', 'user', 'email'],
            where_clauses=where_clauses,
            order_by=order_by,
            offset=offset,
            limit=limit,
        )
        self.assertEqual(rows, [(3, 'user4'), (2, 'user3'), (1, 'user2')])

    def test_db_update_rows(self):
        where_clauses = (WhereClause('uid', '==', 2), WhereClause('user', '==', 'user4'))
        rowcount = self.db.update_rows(
            table_name=self.table_name,
            set_clauses={'Register_datetime': datetime(2025, 2, 1)},
            where_clauses=where_clauses
        )
        self.assertEqual(rowcount, 2)

    def test_db_delete_rows(self):
        where_clauses = (WhereClause('uid', '==', 3), WhereClause('user', '==', 'user5'))
        rowcount = self.db.delete_rows(table_name=self.table_name, where_clauses=where_clauses)
        self.assertEqual(rowcount, 2)

    # Execute SQL
    def test_db_execute_sql(self):
        sql = 'SELECT uid, user AS name FROM users'
        cols, rows = self.db._execute_sql(sql, return_result=True)
        self.assertEqual(cols, ['uid', 'name'])


if __name__ == '__main__':
    unittest.main()
