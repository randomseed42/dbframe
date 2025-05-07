import os
from pathlib import Path
from loguru import logger

import pandas as pd
import pytest
from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.exc import IntegrityError, OperationalError

from dbframe import Order, PG, Where


PG_CONN = dict(
    host='127.0.0.1',
    port=5432,
    user='postgres',
    password='postgres',
    dbname='test_db',
)


@pytest.fixture(scope='session')
def pg():
    DEFAULT_PG_CONN = {**PG_CONN, 'dbname': 'postgres'}
    DEFAULT_PG = PG(**DEFAULT_PG_CONN)
    DEFAULT_PG.create_database('test_db')
    _pg = PG(**PG_CONN, verbose=False)
    _pg.create_schema('test_schema')
    _pg.create_table('test_schema', 'test_table', [Column('id', Integer, primary_key=True), Column('name', String)])
    yield _pg
    _pg.dispose()
    DEFAULT_PG.drop_database('test_db')


class TestPGDatabase:
    def test_get_url(self, pg: PG):
        assert pg.get_url() == 'postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/test_db'

    def test_validate_conn(self, pg: PG):
        assert pg.validate_conn()
        conn_kw = PG_CONN.copy()
        conn_kw.update(dbname='non_existent_db')
        pg2 = PG(**conn_kw)
        pytest.raises(OperationalError, pg2.validate_conn)

    def test_create_database(self, pg: PG):
        dbname = 'Data'
        assert pg.create_database(dbname) == 'data'
        pg.drop_database(dbname)

    def test_get_database(self, pg: PG):
        assert pg.get_database('Test_DB') == 'test_db'

    def test_get_databases(self, pg: PG):
        assert 'test_db' in pg.get_databases()

    def test_drop_database(self, pg: PG):
        dbname = 'Data'
        pg.create_database(dbname)
        assert pg.drop_database(dbname) == 'data'


class TestPGSchema:
    def test_create_schema(self, pg: PG):
        schema_nm = 'Test_Schema_2'
        pg.create_schema(schema_nm)
        assert pg.get_schema(schema_nm) == 'test_schema_2'
        pg.drop_schema(schema_nm)

    def test_get_schema(self, pg: PG):
        assert pg.get_schema('Test_Schema') == 'test_schema'

    def test_get_schemas(self, pg: PG):
        assert 'test_schema' in pg.get_schemas()

    def test_drop_schema(self, pg: PG):
        schema_nm = 'Test_Schema_2'
        pg.create_schema(schema_nm)
        assert pg.drop_schema(schema_nm, cascade=True) == 'test_schema_2'


class TestPGTable:
    def test_create_table(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        assert pg.get_table(schema_nm, tb_nm).name == 'test_table_2'
        pg.drop_table(schema_nm, tb_nm)

    def test_get_table(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table'
        assert pg.get_table(schema_nm, tb_nm).name == 'test_table'

    def test_get_tables(self, pg: PG):
        schema_nm = 'test_schema'
        assert f'{schema_nm}.test_table' in pg.get_tables(schema_nm).keys()

    def test_rename_table(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        new_tb_nm = 'Test_Table_3'
        pg.rename_table(schema_nm, tb_nm, new_tb_nm)
        assert pg.get_table(schema_nm, new_tb_nm).name == 'test_table_3'
        pg.drop_table(schema_nm, new_tb_nm)

    def test_drop_table(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        assert pg.drop_table(schema_nm, tb_nm) == 'test_table_2'

    def test_truncate_table(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        tb = pg.get_table(schema_nm, tb_nm)
        pg._execute_sql(tb.insert().values([{'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True}]))
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 5000.0, True)]
        pg.truncate_table(schema_nm, tb_nm, restart=False)
        assert pg._execute_sql(tb.select()).fetchall() == []
        pg._execute_sql(tb.insert().values([{'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True}]))
        assert pg._execute_sql(tb.select()).fetchall() == [(2, 'Alice', 25, 5000.0, True)]
        pg.truncate_table(schema_nm, tb_nm, restart=True)
        assert pg._execute_sql(tb.select()).fetchall() == []
        pg._execute_sql(tb.insert().values([{'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True}]))
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 5000.0, True)]
        pg.drop_table(schema_nm, tb_nm)


class TestPGColumn:
    def test_add_column(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.add_column(schema_nm, tb_nm, Column('email', String(100)))
        assert pg.get_table(schema_nm, tb_nm).c.keys() == ['id', 'name', 'age', 'salary', 'is_active', 'email']
        pg.drop_table(schema_nm, tb_nm)

    def test_add_columns(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.add_columns(schema_nm, tb_nm, [Column('email', String(100)), Column('phone', String(20))])
        assert pg.get_table(schema_nm, tb_nm).c.keys() == ['id', 'name', 'age', 'salary', 'is_active', 'email', 'phone']
        pg.drop_table(schema_nm, tb_nm)

    def test_get_column(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        assert pg.get_column(schema_nm, tb_nm, 'age').name == 'age'
        pg.drop_table(schema_nm, tb_nm)

    def test_get_columns(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        assert list(pg.get_columns(schema_nm, tb_nm)) == ['id', 'name', 'age', 'salary', 'is_active']
        pg.drop_table(schema_nm, tb_nm)

    def test_rename_column(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.rename_column(schema_nm, tb_nm, 'age', 'age_new')
        assert list(pg.get_columns(schema_nm, tb_nm)) == ['id', 'name', 'age_new', 'salary', 'is_active']
        pg.drop_table(schema_nm, tb_nm)

    def test_alter_column(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.alter_column(schema_nm, tb_nm, 'age', sql_dtype=String(100))
        assert list(pg.get_columns(schema_nm, tb_nm)) == ['id', 'name', 'age', 'salary', 'is_active']
        assert pg.get_column(schema_nm, tb_nm, 'age').type.length == 100
        pg.drop_table(schema_nm, tb_nm)

    def test_drop_column(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.drop_column(schema_nm, tb_nm, 'age')
        assert pg.get_table(schema_nm, tb_nm).c.keys() == ['id', 'name', 'salary', 'is_active']
        pg.drop_table(schema_nm, tb_nm)

    def test_drop_columns(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.drop_columns(schema_nm, tb_nm, ['age', 'salary'])
        assert pg.get_table(schema_nm, tb_nm).c.keys() == ['id', 'name', 'is_active']
        pg.drop_table(schema_nm, tb_nm)


class TestPGIndex:
    def test_create_index(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.create_index(schema_nm, tb_nm, col_nms=['name'], idx_nm='idx_name')
        assert 'idx_name' in pg.get_indexes(schema_nm, tb_nm)
        pg.drop_table(schema_nm, tb_nm)

    def test_get_index(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.create_index(schema_nm, tb_nm, col_nms=['name'], idx_nm='idx_name')
        assert pg.get_index(schema_nm, tb_nm, idx_nm='idx_name').name == 'idx_name'
        pg.drop_table(schema_nm, tb_nm)

    def test_get_indexes(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.create_index(schema_nm, tb_nm, col_nms=['name'], idx_nm='idx_name')
        assert 'idx_name' in pg.get_indexes(schema_nm, tb_nm)
        pg.drop_table(schema_nm, tb_nm)

    def test_drop_index(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.create_index(schema_nm, tb_nm, col_nms=['name'], idx_nm='idx_name')
        assert pg.drop_index(schema_nm, tb_nm, idx_nm='idx_name') == 'idx_name'
        pg.drop_table(schema_nm, tb_nm)


class TestPGRow:
    def test_insert_row(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        tb = pg.create_table(schema_nm, tb_nm, cols)
        pg.insert_row(schema_nm, tb_nm, {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True})
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 5000.0, True)]
        pytest.raises(
            IntegrityError,
            pg.insert_row,
            schema_nm=schema_nm,
            tb_nm=tb_nm,
            row={'id': 1, 'name': 'Alice', 'age': 25, 'salary': 6000.0, 'is_active': False},
        )
        pg.insert_row(
            schema_nm, tb_nm, {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 6000.0, 'is_active': False}, on_conflict='update'
        )
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 6000.0, False)]
        pg.insert_row(
            schema_nm,
            tb_nm,
            {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
            on_conflict='do_nothing',
        )
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 6000.0, False)]
        pg.drop_table(schema_nm, tb_nm)

    def test_insert_rows(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        tb = pg.create_table(schema_nm, tb_nm, cols)
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ],
        )
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 5000.0, True), (2, 'Bob', 30, 6000.0, False)]
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 6000.0, 'is_active': False},
                {'id': 2, 'name': 'Bob', 'age': 30, 'salary': 5000.0, 'is_active': True},
            ],
            on_conflict='update',
        )
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 6000.0, False), (2, 'Bob', 30, 5000.0, True)]
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'id': 2, 'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ],
            on_conflict='do_nothing',
        )
        assert pg._execute_sql(tb.select()).fetchall() == [(1, 'Alice', 25, 6000.0, False), (2, 'Bob', 30, 5000.0, True)]
        pg.drop_table(schema_nm, tb_nm)

    def test_select_rows(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ],
        )
        assert pg.select_rows(schema_nm, tb_nm) == (
            ['id', 'name', 'age', 'salary', 'is_active'],
            [(1, 'Alice', 25, 5000.0, True), (2, 'Bob', 30, 6000.0, False)],
        )
        assert pg.select_rows(schema_nm, tb_nm, col_nms=['name', 'age']) == (['name', 'age'], [('Alice', 25), ('Bob', 30)])
        assert pg.select_rows(schema_nm, tb_nm, col_nms=['name', 'salary'], where=Where('age', '>', 25)) == (
            ['name', 'salary'],
            [('Bob', 6000.0)],
        )
        assert pg.select_rows(
            schema_nm, tb_nm, col_nms=['name', 'age'], where=(Where('age', '>', 25), Where('is_active', '==', True))
        ) == (['name', 'age'], [('Alice', 25), ('Bob', 30)])
        assert pg.select_rows(
            schema_nm, tb_nm, col_nms=['name', 'age'], where=[Where('age', '>', 25), Where('is_active', '==', True)]
        ) == (['name', 'age'], [])
        assert pg.select_rows(
            schema_nm,
            tb_nm,
            col_nms=['name', 'age'],
            where=(Where('age', '>', 25), Where('is_active', '==', True)),
            order=Order('age', ascending=False),
        ) == (['name', 'age'], [('Bob', 30), ('Alice', 25)])
        assert pg.select_rows(
            schema_nm,
            tb_nm,
            col_nms=['name', 'age'],
            where=(Where('age', '>', 25), Where('is_active', '==', True)),
            order=Order('age', ascending=False),
            limit=1,
        ) == (['name', 'age'], [('Bob', 30)])
        assert pg.select_rows(
            schema_nm,
            tb_nm,
            col_nms=['name', 'age'],
            where=(Where('age', '>', 25), Where('is_active', '==', True)),
            order=Order('age', ascending=False),
            limit=1,
            offset=1,
        ) == (['name', 'age'], [('Alice', 25)])
        pg.drop_table(schema_nm, tb_nm)

    def test_update_rows(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ],
        )
        pg.update_rows(
            schema_nm,
            tb_nm,
            {'name': 'Charlie', 'age': 35, 'is_active': True},
            where=Where('name', '==', 'Bob'),
        )
        assert pg.select_rows(schema_nm, tb_nm, order=Order('id')) == (
            ['id', 'name', 'age', 'salary', 'is_active'],
            [(1, 'Alice', 25, 5000.0, True), (2, 'Charlie', 35, 6000.0, True)],
        )
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'name': 'Dave', 'age': 40, 'salary': 6000.0, 'is_active': False},
            ],
        )
        pg.update_rows(
            schema_nm,
            tb_nm,
            {'salary': 5500.0},
            where=Where('age', '<', 35),
        )
        assert pg.select_rows(schema_nm, tb_nm, order=Order('id')) == (
            ['id', 'name', 'age', 'salary', 'is_active'],
            [(1, 'Alice', 25, 5500.0, True), (2, 'Charlie', 35, 6000.0, True), (3, 'Dave', 40, 6000.0, False)],
        )
        pg.drop_table(schema_nm, tb_nm)

    def test_delete_rows(self, pg: PG):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer),
            Column('salary', Float),
            Column('is_active', Boolean),
        ]
        pg.create_table(schema_nm, tb_nm, cols)
        pg.insert_rows(
            schema_nm,
            tb_nm,
            [
                {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ],
        )
        pg.delete_rows(schema_nm, tb_nm, where=Where('name', '==', 'Bob'))
        assert pg.select_rows(schema_nm, tb_nm, order=Order('id')) == (
            ['id', 'name', 'age', 'salary', 'is_active'],
            [(1, 'Alice', 25, 5000.0, True)],
        )
        pg.drop_table(schema_nm, tb_nm)

    def test_execute_sql(self, pg: PG):
        schema_nm = 'test_schema'
        pg.create_table(schema_nm=schema_nm, tb_nm='test_table_2', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 1, 'name': 'Alice'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 2, 'name': 'Bob'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 3, 'name': 'Charlie'})
        pg.create_table(schema_nm=schema_nm, tb_nm='test_table_3', cols=[Column('id', Integer, primary_key=True), Column('age', Integer)])
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 1, 'age': 20})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 2, 'age': 30})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 3, 'age': 40})
        assert pg._execute_sql(f'SELECT * FROM {schema_nm}.test_table_2').fetchall() == [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
        assert pg._execute_sql(f'SELECT t2.name, t3.age FROM {schema_nm}.test_table_2 t2 JOIN {schema_nm}.test_table_3 t3 ON t2.id = t3.id').fetchall() == [
            ('Alice', 20),
            ('Bob', 30),
            ('Charlie', 40),
        ]
