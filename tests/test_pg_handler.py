import os
import unittest
from datetime import datetime
from logging import FileHandler

import pandas as pd
from dbframe import PGDFHandler
from dbframe.utils import NamingValidator, OrderByClause, WhereClause
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.exc import IntegrityError


class TestPGHandler(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='PG_Logger',
        log_level='CRITICAL',
        log_console=True,
        log_file='pg.log',
    )

    @staticmethod
    def _generate_columns() -> list[Column]:
        return [
            Column('uid', Integer, primary_key=True),
            Column('user', String),
            Column('register_datetime', DateTime),
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
        cls.db.create_table(schema=cls.schema, table_name=cls.table_name, columns=cls.columns)
        cls.db.insert_rows(schema=cls.schema, table_name=cls.table_name, rows=rows, on_conflict='do_nothing')

    @classmethod
    def _drop_table(cls):
        cls.db.drop_table(schema=cls.schema, table_name=cls.table_name)

    @classmethod
    def setUpClass(cls):
        cls.default_db_conf = dict(
            host='127.0.0.1',
            port=5432,
            user='postgres',
            password='postgres',
            dbname='postgres',
        )
        cls.dbname = 'test_DataBase'
        cls.db_conf = cls.default_db_conf.copy()
        cls.db_conf.update({'dbname': cls.dbname})
        cls.schema = 'test_SchemA'
        cls.table_name = 'users'
        cls.default_db = PGDFHandler(**cls.default_db_conf, **cls.LOGGER_CONF)
        cls.default_db.create_database(dbname=cls.dbname)
        cls.db = PGDFHandler(**cls.db_conf, **cls.LOGGER_CONF)
        cls.db.create_schema(schema=cls.schema)

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_table(table_name=cls.table_name, schema=cls.schema)
        cls.db.drop_schema(schema=cls.schema)
        cls.db.engine.dispose()
        cls.default_db.drop_database(dbname=cls.dbname)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def setUp(self):
        self._create_table()

    def tearDown(self):
        self._drop_table()

    def test_db_init(self):
        databases = self.db.get_databases()
        self.assertTrue(NamingValidator.dbname(self.db_conf.get('dbname')) in databases)

    # Database CRUD
    def test_db_create_database(self):
        temp_dbname = 'Temp_DB'
        dbname = self.db.create_database(dbname=temp_dbname)
        databases = self.db.get_databases()
        self.assertTrue(NamingValidator.dbname(dbname) in databases)
        self.db.drop_database(dbname=temp_dbname)

        dbname = self.db.create_database(dbname='postgres')
        databases = self.db.get_databases()
        self.assertTrue(dbname is None and 'postgres' in databases)

    def test_db_get_database(self):
        temp_dbname = 'tEMp_dB'
        none_temp_dbname = self.db.get_database(dbname=temp_dbname)
        self.assertEqual(none_temp_dbname, None)
        temp_dbname2 = self.db.create_database(dbname=temp_dbname)
        temp_dbname3 = self.db.get_database(dbname=temp_dbname)
        self.assertEqual(temp_dbname2, temp_dbname3)
        self.db.drop_database(dbname=temp_dbname)

    def test_db_drop_database(self):
        temp_dbname = 'TeMp_Db'
        temp_dbname2 = self.db.create_database(dbname=temp_dbname)
        temp_dbname3 = self.db.drop_database(dbname=temp_dbname)
        self.assertEqual(temp_dbname2, temp_dbname3)

    # Schema CRUD
    def test_db_create_drop_schema(self):
        temp_schema = 'tEMp_sCHema'
        temp_schema2 = self.db.create_schema(schema=temp_schema)
        schemas = self.db.get_schemas()
        self.assertEqual(temp_schema2, NamingValidator.schema(temp_schema))
        self.assertTrue(temp_schema2 in schemas)
        try:
            temp_schema3 = self.db.drop_schema(schema=temp_schema, cascade=False)
            schemas = self.db.get_schemas()
        except Exception as err:
            self.db.logger.critical(err)
            temp_schema3 = self.db.drop_schema(schema=temp_schema, cascade=True)
        self.assertEqual(temp_schema3, NamingValidator.schema(temp_schema))
        self.assertTrue(temp_schema3 not in schemas)

    def test_db_get_schema(self):
        schema = self.db.get_schema(schema=self.schema)
        self.assertEqual(schema, NamingValidator.schema(schema))

    def test_db_get_schemas(self):
        default_schemas = {'pg_toast', 'pg_catalog', 'public', 'information_schema'}
        schemas = self.db.get_schemas()
        self.assertTrue(default_schemas.issubset(schemas))

    # Table CRUD
    def test_db_create_table(self):
        temp_table = 'tEmP_tAbLe'
        table = self.db.create_table(schema=self.schema, table_name=temp_table, columns=self._generate_columns())
        self.assertEqual(table.name, NamingValidator.table(temp_table))
        self.db.drop_table(schema=self.schema, table_name=temp_table)

        with self.assertRaisesRegex(ValueError, 'Table.*already exists.*'):
            self.db.create_table(schema=self.schema, table_name=self.table_name, columns=self._generate_columns())

        with self.assertRaisesRegex(ValueError, 'Columns.*cannot be empty'):
            temp_table2 = 'temp_table2'
            self.db.create_table(schema=self.schema, table_name=temp_table2, columns=[])

    def test_db_get_table(self):
        table = self.db.get_table(schema='pg_catalog', table_name='pg_roles')
        self.assertTrue(table.schema == 'pg_catalog' and table.name == 'pg_roles')

        table = self.db.get_table(schema='public', table_name='pg_roles')
        self.assertEqual(table, None)

    def test_db_get_tables(self):
        default_tables = {
            'information_schema.sql_features',
            'information_schema.sql_sizing',
            'information_schema.sql_parts',
            'information_schema.sql_implementation_info',
        }
        tables = self.db.get_tables(schema='information_schema')
        self.assertTrue(default_tables.issubset(tables.keys()))

        tables = self.db.get_tables(schema='information_schema', views=True)
        self.assertTrue('information_schema.schemata' in tables)

    def test_db_create_rename_drop_table(self):
        temp_table = 'Temp_Table'
        self.db.create_table(schema=self.schema, table_name=temp_table, columns=self._generate_columns())
        self.assertTrue(self.db.get_table(schema=self.schema, table_name=temp_table) is not None)

        new_temp_table = 'new_tEMP_tAble'
        self.db.rename_table(schema=self.schema, old_table_name=temp_table, new_table_name=new_temp_table)
        self.assertTrue(self.db.get_table(schema=self.schema, table_name=temp_table) is None)
        self.assertTrue(self.db.get_table(schema=self.schema, table_name=new_temp_table) is not None)

        self.db.drop_table(schema=self.schema, table_name=new_temp_table)
        table = self.db.get_table(schema=self.schema, table_name=temp_table)
        self.assertEqual(table, None)

        with self.assertRaisesRegex(ValueError, 'Table.*already exists.*'):
            self.db.create_table(schema=self.schema, table_name='users', columns=self._generate_columns())

    # Column CRUD
    def test_db_add_column(self):
        new_column = Column('aGe', Integer)
        self.db.add_column(schema=self.schema, table_name=self.table_name, column=new_column)
        self.assertTrue('age' in self.db.get_columns(schema=self.schema, table_name=self.table_name))

    def test_db_add_columns(self):
        new_columns = [
            Column('uid', Integer),
            Column('first_NaMe', String),
            Column('last_name', String),
        ]
        self.db.add_columns(schema=self.schema, table_name=self.table_name, columns=new_columns)
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue('first_name' in current_columns and 'last_name' in current_columns)

    def test_db_get_column(self):
        column = self.db.get_column(schema=self.schema, table_name=self.table_name, column_name='user')
        self.assertEqual(column.name, 'user')
        column2 = self.db.get_column(schema=self.schema, table_name=self.table_name, column_name='user2')
        self.assertEqual(column2, None)

    def test_db_get_columns(self):
        columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue(set(col.name for col in self.columns).issubset(columns.keys()))

    def test_db_alter_column(self):
        column_name = self.db.alter_column(schema=self.schema, table_name=self.table_name, old_column_name='user',
                                           new_column_name='username', sql_dtype=None)
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue(column_name in current_columns and 'user' not in current_columns)
        ...

    def test_db_drop_column(self):
        new_column = Column('email', String)
        self.db.add_column(schema=self.schema, table_name=self.table_name, column=new_column)
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue('email' in current_columns)
        self.db.drop_column(schema=self.schema, table_name=self.table_name, column_name='email')
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue('email' not in current_columns)

    def test_db_drop_columns(self):
        new_columns = [
            Column('password', String),
            Column('role', String),
        ]
        self.db.add_columns(schema=self.schema, table_name=self.table_name, columns=new_columns)
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue('password' in current_columns and 'role' in current_columns)
        self.db.drop_columns(schema=self.schema, table_name=self.table_name, column_names=['password', 'role'])
        current_columns = self.db.get_columns(schema=self.schema, table_name=self.table_name)
        self.assertTrue('password' not in current_columns and 'role' not in current_columns)

    # Index CRUD
    def test_db_create_index(self):
        self.db.create_index(schema=self.schema, table_name=self.table_name, column_names=['uid', 'user'])
        indexes = self.db.get_indexes(schema=self.schema, table_name=self.table_name)
        self.assertTrue(f'ix_{NamingValidator.table(self.table_name)}_uid_user' in indexes)
        with self.assertRaisesRegex(ValueError, 'Index .* already exists'):
            self.db.create_index(schema=self.schema, table_name=self.table_name, column_names=['uid', 'user'])
        with self.assertRaisesRegex(ValueError, 'Column .* not in'):
            self.db.create_index(schema=self.schema, table_name=self.table_name, column_names=['user', 'email'])

    def test_db_get_indexes(self):
        indexes = self.db.get_indexes(schema=self.schema, table_name=self.table_name)
        self.assertEqual(len(indexes), 0)
        self.db.create_index(schema=self.schema, table_name=self.table_name, column_names=['uid', 'user'])
        indexes = self.db.get_indexes(schema=self.schema, table_name=self.table_name)
        self.assertEqual(indexes, {f'ix_{NamingValidator.table(self.table_name)}_uid_user': ['uid', 'user']})

    def test_db_drop_index(self):
        self.db.create_index(schema=self.schema, table_name=self.table_name, column_names=['uid', 'user'])
        idx_name = self.db.drop_index(schema=self.schema, table_name=self.table_name, column_names=['uid', 'user'])
        self.assertEqual(f'ix_{NamingValidator.table(self.table_name)}_uid_user', idx_name)
        indexes = self.db.get_indexes(schema=self.schema, table_name=self.table_name)
        self.assertTrue(idx_name not in indexes)

    # Rows CRUD
    def test_db_insert_rows(self):
        rows = [
            dict(uid=5, user='user6', register_datetime=datetime(2025, 1, 6, 0, 0, 0)),
            dict(uid=6, user='user7', register_datetime=datetime(2025, 1, 7, 0, 0, 0)),
        ]
        rowcount = self.db.insert_rows(schema=self.schema, table_name=self.table_name, rows=rows,
                                       on_conflict='do_nothing')
        self.assertEqual(rowcount, len(rows))

    def test_db_select_rows(self):
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name)
        self.assertEqual(cols, ['uid', 'user', 'register_datetime'])
        self.assertEqual(len(rows), 5)

        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'])
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(rows, [(0, 'user1'), (1, 'user2'), (2, 'user3'), (3, 'user4'), (4, 'user5')])

        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name, column_names=['email'])
        self.assertEqual((cols, rows), (None, None))

    def test_db_select_rows_where(self):
        where_clauses = WhereClause('uid', '<', 2)
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 2)

        where_clauses = [WhereClause('uid', '>=', 2)]
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

        where_clauses = ([WhereClause('uid', '>', 0), WhereClause('uid', '<', 2)], WhereClause('uid', '>', 2))
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 4)

        where_clauses = (WhereClause('User', 'like', 'User1'), WhereClause('USER', 'ilike', 'User2'),
                         WhereClause('user', 'ilike', 'user3'), WhereClause('user', 'ilike', '%SER4'))
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses)
        self.assertEqual(cols, ['uid', 'user'])
        self.assertEqual(len(rows), 3)

    def test_db_select_rows_where_order(self):
        where_clauses = (WhereClause('uid', 'between', (0, 2)), WhereClause('uid', '>', 3))
        order_by = [OrderByClause('uid', False)]
        cols, rows = self.db.select_rows(schema=self.schema, table_name=self.table_name,
                                         column_names=['uid', 'user', 'email'], where_clauses=where_clauses,
                                         order_by=order_by)
        self.assertEqual(rows, [(4, 'user5'), (2, 'user3'), (1, 'user2'), (0, 'user1')])

    def test_db_select_rows_where_order_offset_limit(self):
        where_clauses = None
        order_by = [OrderByClause('uid', False)]
        offset = 1
        limit = 3
        cols, rows = self.db.select_rows(
            schema=self.schema,
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
            schema=self.schema,
            table_name=self.table_name,
            set_clauses={'Register_datetime': datetime(2025, 2, 1)},
            where_clauses=where_clauses
        )
        self.assertEqual(rowcount, 2)

    def test_db_delete_rows(self):
        where_clauses = (WhereClause('uid', '==', 3), WhereClause('user', '==', 'user5'))
        rowcount = self.db.delete_rows(schema=self.schema, table_name=self.table_name, where_clauses=where_clauses)
        self.assertEqual(rowcount, 2)

    # Execute SQL
    def test_db_execute_sql(self):
        sql = 'SELECT uid, user AS name FROM test_schema.users'
        cols, rows = self.db._execute_sql(sql, return_result=True)
        self.assertEqual(cols, ['uid', 'name'])


class TestPGDFHandler(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='PG_DF_Logger',
        log_level='CRITICAL',
        log_console=True,
        log_file='pg.log',
    )

    @staticmethod
    def _create_df():
        data = [
            ['John', 17, 1.75, '2025-01-01 01:00:00'],
            ['Jack', 18, 1.80, '2025-01-02 02:00:00'],
            ['Jane', 19, 1.66, '2025-01-03 03:00:00'],
            ['Judy', 16, 1.62, '2025-01-04 04:00:00'],
        ]
        columns = ['nAmE', 'aGe', 'height', 'log_datetime']
        df = pd.DataFrame(data=data, columns=columns)
        return df

    @classmethod
    def setUpClass(cls):
        cls.default_db_conf = dict(
            host='127.0.0.1',
            port=5432,
            user='postgres',
            password='postgres',
            dbname='postgres',
        )
        cls.dbname = 'test_DataBase'
        cls.db_conf = cls.default_db_conf.copy()
        cls.db_conf.update({'dbname': cls.dbname})
        cls.schema = 'test_SchemA'
        cls.table_name = 'users'
        cls.default_db = PGDFHandler(**cls.default_db_conf, **cls.LOGGER_CONF)
        cls.default_db.create_database(dbname=cls.dbname)
        cls.db = PGDFHandler(**cls.db_conf, **cls.LOGGER_CONF)
        cls.db.create_schema(schema=cls.schema)

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_table(table_name=cls.table_name, schema=cls.schema)
        cls.db.drop_schema(schema=cls.schema)
        cls.db.engine.dispose()
        cls.default_db.drop_database(dbname=cls.dbname)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def setUp(self):
        self.df = self._create_df()

    def tearDown(self):
        self.db.drop_table(schema=self.schema, table_name='temp')

    def test_df_create_table(self):
        temp_table_name = 'tEmP'

        # Create 1
        table = self.db.df_create_table(df=self.df, schema=self.schema, table_name=temp_table_name)
        self.assertEqual(table.name, NamingValidator.table(temp_table_name))
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

        # Create 2
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='MY_uid',
        )
        self.assertEqual(table.name, NamingValidator.table(temp_table_name))
        self.assertEqual(table.primary_key.columns[0].name, NamingValidator.column('MY_uid'))
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

        # Create 3
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='MY_uid',
            notnull_column_names=['nAmE', 'aGe'],
        )
        self.assertEqual(table.name, NamingValidator.table(temp_table_name))
        self.assertFalse(table.columns.get('name').nullable)
        self.assertFalse(table.columns[2].nullable)
        self.assertTrue(table.columns[3].nullable)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

        # Create 4
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='MY_uid',
            notnull_column_names=['nAmE', 'aGe'],
            index_column_names=['nAmE', ['aGe', 'height']],
        )
        self.assertEqual(table.name, NamingValidator.table(temp_table_name))
        indexes = [idx.name for idx in table.indexes]
        self.assertTrue('ix_temp_name' in indexes)
        self.assertTrue('ix_temp_age_height' in indexes)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

        # Create 5
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='MY_uid',
            notnull_column_names=['nAmE', 'aGe'],
            index_column_names=['nAmE', ['aGe', 'height']],
            unique_column_names=['nAmE', ['aGe', 'height']],
        )
        self.assertEqual(table.name, NamingValidator.table(temp_table_name))
        constraints = [c.name for c in table.constraints]
        self.assertTrue('uix_temp_name' in constraints)
        self.assertTrue('uix_temp_age_height' in constraints)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

    def test_df_add_columns(self):
        temp_table_name = 'tEmP'
        table = self.db.df_create_table(df=self.df, schema=self.schema, table_name=temp_table_name)
        data = [
            ['John', 17, 1.75, 70],
            ['Jack', 18, 1.80, 80],
            ['Jane', 19, 1.66, 60],
            ['Judy', 16, 1.62, 50],
        ]
        columns = ['nAmE', 'aGe', 'height', 'weight']
        df = pd.DataFrame(data=data, columns=columns)
        added_column_names = self.db.df_add_columns(df=df, schema=self.schema, table_name=temp_table_name)
        self.assertEqual(added_column_names, ['weight'])
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

    def test_df_alter_columns_type(self):
        temp_table_name = 'tEmP'
        table = self.db.df_create_table(df=self.df, schema=self.schema, table_name=temp_table_name)
        data = [
            ['John', 17.8, 1.75, 70],
            ['Jack', 18, 1.80, 80],
            ['Jane', 19, 1.66, 60],
            ['Judy', 16, 1.62, 50],
        ]
        columns = ['nAmE', 'AgE', 'height', 'weight']
        df = pd.DataFrame(data=data, columns=columns)
        self.db.df_alter_columns_type(df=df, schema=self.schema, table_name=temp_table_name)
        current_columns = self.db.get_columns(schema=self.schema, table_name=temp_table_name)
        self.assertIsInstance(current_columns['age'].type, Float)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

    def test_df_insert_rows(self):
        # Case 1
        temp_table_name = 'tEmP'
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='uid',
            notnull_column_names=['nAmE', 'aGe'],
            index_column_names=['nAmE', ['aGe', 'height']],
            unique_column_names=['nAmE', ['aGe', 'height']],
        )

        data = [
            ['Joan', 17.7, 1.75, 70],
            ['Jess', 18.1, 1.80, 80],
            ['June', 16.0, 1.62, 60],
            ['July', 19.2, 1.66, 60],
            ['Jake', 19.3, 1.68, None],
        ]
        columns = ['nAmE', 'AgE', 'height', 'weight']
        df = pd.DataFrame(data=data, columns=columns)
        rowcount = self.db.df_insert_rows(df=df, schema=self.schema, table_name=temp_table_name, on_conflict='do_nothing')
        self.assertEqual(rowcount, 4)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)

        # Case 2
        temp_table_name = 'tEmP'
        table = self.db.df_create_table(
            df=self.df, schema=self.schema, table_name=temp_table_name,
            primary_column_name='index', primary_sql_column_name='uid',
            notnull_column_names=['nAmE', 'aGe'],
            index_column_names=['nAmE', ['aGe', 'height']],
            unique_column_names=['nAmE', ['aGe', 'height']],
        )

        data = [
            ['Joan', 17.7, 1.75, '70'],
            ['Jess', 18.1, 1.80, '80'],
            ['June', 16.0, 1.62, '60'],
            ['July', 19.2, 1.66, '60'],
            ['Jake', 19.3, 1.68, None],
        ]
        columns = ['nAmE', 'AgE', 'height', 'weight']
        df = pd.DataFrame(data=data, columns=columns)
        with self.assertRaisesRegex(IntegrityError, 'duplicate key value violates unique constraint'):
            rowcount = self.db.df_insert_rows(df=df, schema=self.schema, table_name=temp_table_name, on_conflict=None)
            self.db.logger.critical(rowcount)
        self.db.drop_table(schema=self.schema, table_name=temp_table_name)


if __name__ == '__main__':
    unittest.main()
