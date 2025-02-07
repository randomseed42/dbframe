import os
import unittest
from datetime import datetime
from logging import FileHandler

from sqlalchemy import Column, TEXT, INTEGER, TIMESTAMP

from dbframe import PGDFHandler
from dbframe.utils import WhereClause, OrderByClause


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
        cls.db.create_table(table_name=cls.table_name, columns=cls.columns, schema=cls.schema)
        # cls.db.insert_rows(table_name=cls.table_name, rows=rows, on_conflict='do_nothing')

    @classmethod
    def setUpClass(cls):
        cls.default_db_conf = dict(
            host='127.0.0.1',
            port=5432,
            user='postgres',
            password='postgres',
            dbname='postgres',
        )
        cls.dbname = 'test_database'
        cls.db_conf = cls.default_db_conf.copy()
        cls.db_conf.update({'dbname': cls.dbname})
        cls.schema = 'test_schema'
        cls.table_name = 'users'

        cls.default_db = PGDFHandler(**cls.default_db_conf, **cls.LOGGER_CONF)
        cls.default_db.create_database(dbname=cls.dbname)

        cls.db = PGDFHandler(**cls.db_conf, **cls.LOGGER_CONF)
        cls.db.create_schema(schema=cls.schema)
        cls._create_table()

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_table(table_name=cls.table_name, schema=cls.schema)
        for hdlr in cls.db.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    def test_db_init(self):
        databases = self.db.get_databases()
        self.assertTrue(self.db_conf.get('dbname') in databases)

    def test_db_create_drop_database(self):
        temp_dbname = 'temp_db'
        dbname = self.db.create_database(dbname=temp_dbname)
        databases = self.db.get_databases()
        self.assertTrue(dbname in databases)

        dbname = self.db.drop_database(dbname=temp_dbname)
        databases = self.db.get_databases()
        self.assertTrue(dbname not in databases)

        dbname = self.db.create_database(dbname='postgres')
        databases = self.db.get_databases()
        self.assertTrue(dbname is None and 'postgres' in databases)

    def test_db_get_schemas(self):
        default_schemas = {'pg_toast', 'pg_catalog', 'public', 'information_schema'}
        schemas = self.db.get_schemas()
        self.assertTrue(default_schemas.issubset(schemas))

    def test_db_create_drop_schema(self):
        temp_schema = 'temp_schema'
        schema = self.db.create_schema(schema=temp_schema)
        schemas = self.db.get_schemas()
        self.assertTrue(schema in schemas)

        schema = self.db.drop_schema(schema=temp_schema)
        schemas = self.db.get_schemas()
        self.assertTrue(schema not in schemas)

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

    def test_db_get_table(self):
        table = self.db.get_table(table_name='pg_roles', schema='pg_catalog')
        self.assertTrue(table.name == 'pg_roles' and table.schema == 'pg_catalog')

        table = self.db.get_table(table_name='pg_roles', schema='public')
        self.assertEqual(table, None)

    def test_db_create_rename_drop_table(self):
        temp_table = 'temp_table'
        table = self.db.create_table(table_name=temp_table, schema=self.schema, columns=self._generate_columns())
        self.assertTrue(self.db.get_table(table_name=temp_table, schema=self.schema) is not None)

        new_temp_table = 'new_temp_table'
        self.db.rename_table(old_table_name=temp_table, new_table_name=new_temp_table, schema=self.schema)
        self.assertTrue(self.db.get_table(table_name=temp_table, schema=self.schema) is None)
        self.assertTrue(self.db.get_table(table_name=new_temp_table, schema=self.schema) is not None)

        self.db.drop_table(table_name=new_temp_table, schema=self.schema)
        table = self.db.get_table(table_name=temp_table, schema=self.schema)
        self.assertEqual(table, None)

        with self.assertRaisesRegex(ValueError, 'Table.*already exists.*'):
            self.db.create_table(table_name='users', columns=self._generate_columns(), schema=self.schema)


if __name__ == '__main__':
    unittest.main()
