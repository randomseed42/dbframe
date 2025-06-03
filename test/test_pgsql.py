import pandas as pd
import pytest
from psycopg2.errors import UniqueViolation
from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from dbframe import Order, PgsqlDF, Where

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
    DEFAULT_PG = PgsqlDF(**DEFAULT_PG_CONN)
    DEFAULT_PG.create_database('test_db')
    _pg = PgsqlDF(**PG_CONN, verbose=True)
    _pg.create_schema('test_schema')
    _pg.create_table('test_schema', 'test_table', [Column('id', Integer, primary_key=True), Column('name', String)])
    yield _pg
    _pg.dispose()
    DEFAULT_PG.drop_database('test_db')


class TestPgsqlDatabase:
    def test_get_url(self, pg: PgsqlDF):
        assert pg.get_url() == 'postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/test_db'

    def test_validate_conn(self, pg: PgsqlDF):
        assert pg.validate_conn()
        conn_kw = PG_CONN.copy()
        conn_kw.update(dbname='non_existent_db')
        pg2 = PgsqlDF(**conn_kw)
        pytest.raises(OperationalError, pg2.validate_conn)

    def test_create_database(self, pg: PgsqlDF):
        dbname = 'Test_DB2'
        assert pg.create_database(dbname) == 'test_db2'
        pg.drop_database(dbname)

    def test_get_database(self, pg: PgsqlDF):
        assert pg.get_database('Test_DB') == 'test_db'

    def test_get_databases(self, pg: PgsqlDF):
        assert 'test_db' in pg.get_databases()

    def test_drop_database(self, pg: PgsqlDF):
        dbname = 'Test_DB2'
        pg.create_database(dbname)
        assert pg.drop_database(dbname) == 'test_db2'


class TestPgsqlSchema:
    def test_create_schema(self, pg: PgsqlDF):
        schema_nm = 'Test_Schema_2'
        pg.create_schema(schema_nm)
        assert pg.get_schema(schema_nm) == 'test_schema_2'
        pg.drop_schema(schema_nm)

    def test_get_schema(self, pg: PgsqlDF):
        assert pg.get_schema('Test_Schema') == 'test_schema'

    def test_get_schemas(self, pg: PgsqlDF):
        assert 'test_schema' in pg.get_schemas()

    def test_drop_schema(self, pg: PgsqlDF):
        schema_nm = 'Test_Schema_2'
        pg.create_schema(schema_nm)
        assert pg.drop_schema(schema_nm, cascade=True) == 'test_schema_2'


class TestPgsqlTable:
    def test_create_table(self, pg: PgsqlDF):
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
        pytest.raises(ValueError, pg.create_table, schema_nm, tb_nm, cols)
        pg.drop_table(schema_nm, tb_nm)

    def test_table_exists(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table'
        assert pg.table_exists(schema_nm, tb_nm)
        assert not pg.table_exists(schema_nm, 'non_existent_table')
        assert not pg.table_exists('non_existent_schema', 'test_table')

    def test_get_table(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table'
        assert pg.get_table(schema_nm, tb_nm).name == 'test_table'

    def test_get_tables(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        assert f'{schema_nm}.test_table' in pg.get_tables(schema_nm).keys()

    def test_rename_table(self, pg: PgsqlDF):
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

    def test_drop_table(self, pg: PgsqlDF):
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

    def test_truncate_table(self, pg: PgsqlDF):
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


class TestPgsqlColumn:
    def test_add_column(self, pg: PgsqlDF):
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
        pytest.raises(ProgrammingError, pg.add_column, schema_nm, tb_nm, Column('email', String(100)))
        pg.drop_table(schema_nm, tb_nm)

    def test_add_columns(self, pg: PgsqlDF):
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

    def test_get_column(self, pg: PgsqlDF):
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

    def test_get_columns(self, pg: PgsqlDF):
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

    def test_rename_column(self, pg: PgsqlDF):
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
        pytest.raises(ProgrammingError, pg.rename_column, schema_nm, tb_nm, 'age_new', 'id')
        pg.drop_table(schema_nm, tb_nm)

    def test_alter_column(self, pg: PgsqlDF):
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
        pg.alter_column(schema_nm, tb_nm, 'name', sql_dtype=Integer())
        pg.drop_table(schema_nm, tb_nm)

    def test_drop_column(self, pg: PgsqlDF):
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

    def test_drop_columns(self, pg: PgsqlDF):
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


class TestPgsqlIndex:
    def test_create_index(self, pg: PgsqlDF):
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

    def test_get_index(self, pg: PgsqlDF):
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

    def test_get_indexes(self, pg: PgsqlDF):
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

    def test_drop_index(self, pg: PgsqlDF):
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


class TestPgsqlRow:
    def test_insert_row(self, pg: PgsqlDF):
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

    def test_insert_rows(self, pg: PgsqlDF):
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

    def test_select_rows(self, pg: PgsqlDF):
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

    def test_update_rows(self, pg: PgsqlDF):
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

    def test_delete_rows(self, pg: PgsqlDF):
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

    def test_execute_sql(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        pg.create_table(
            schema_nm=schema_nm, tb_nm='test_table_2', cols=[Column('id', Integer, primary_key=True), Column('name', String)]
        )
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 1, 'name': 'Alice'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 2, 'name': 'Bob'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 3, 'name': 'Charlie'})
        pg.create_table(
            schema_nm=schema_nm, tb_nm='test_table_3', cols=[Column('id', Integer, primary_key=True), Column('age', Integer)]
        )
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 1, 'age': 20})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 2, 'age': 30})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 3, 'age': 40})
        assert pg._execute_sql(f'SELECT * FROM {schema_nm}.test_table_2').fetchall() == [
            (1, 'Alice'),
            (2, 'Bob'),
            (3, 'Charlie'),
        ]
        assert pg._execute_sql(
            f'SELECT t2.name, t3.age FROM {schema_nm}.test_table_2 t2 JOIN {schema_nm}.test_table_3 t3 ON t2.id = t3.id'
        ).fetchall() == [
            ('Alice', 20),
            ('Bob', 30),
            ('Charlie', 40),
        ]
        pg.drop_table(schema_nm, 'test_table_2')
        pg.drop_table(schema_nm, 'test_table_3')


class TestPgsqlDF:
    def test_df_create_table(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        pg.df_create_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        assert f'{schema_nm}.test_table_2' in pg.get_tables(schema_nm=schema_nm).keys()
        assert list(pg.get_columns(schema_nm=schema_nm, tb_nm='test_table_2').keys()) == ['id', 'name']
        pg.drop_table(schema_nm, tb_nm)

    def test_df_add_columns(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        pg.df_create_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        df2 = pd.DataFrame({'id': [1, 2, 3], 'age': [30, 40, 50]})
        new_col_nms = pg.df_add_columns(df=df2, schema_nm=schema_nm, tb_nm=tb_nm)
        assert new_col_nms == ['age']
        assert list(pg.get_columns(schema_nm=schema_nm, tb_nm='test_table_2').keys()) == ['id', 'name', 'age']
        pg.drop_table(schema_nm, tb_nm)

    def test_df_alter_columns(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame.from_records(
            [
                {'name': 'Alice', 'age': 25, 'salary': 5000.0, 'is_active': True},
                {'name': 'Bob', 'age': 30, 'salary': 6000.0, 'is_active': False},
            ]
        )
        pg.df_create_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
        assert isinstance(pg.get_column(schema_nm=schema_nm, tb_nm='test_table_2', col_nm='salary').type, Integer)
        df2 = pd.DataFrame.from_records(
            [
                {'name': 'Charlie', 'age': 45, 'salary': 7000.5, 'is_active': True},
            ]
        )
        pg.df_alter_columns(df=df2, schema_nm=schema_nm, tb_nm=tb_nm)
        assert isinstance(pg.get_column(schema_nm=schema_nm, tb_nm='test_table_2', col_nm='salary').type, Float)
        pg.drop_table(schema_nm, tb_nm)

    def test_df_copy_rows(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        pg.df_create_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='index')
        res = pg.df_copy_rows(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
        assert res == 3
        df2 = pd.DataFrame({'name': ['Dave', 'Eve', 'Frank']})
        res = pg.df_copy_rows(df=df2, schema_nm=schema_nm, tb_nm=tb_nm)
        assert pg.select_rows(schema_nm, tb_nm) == (
            ['uid', 'id', 'name'],
            [(1, 1, 'Alice'), (2, 2, 'Bob'), (3, 3, 'Charlie'), (4, None, 'Dave'), (5, None, 'Eve'), (6, None, 'Frank')],
        )
        df3 = pd.DataFrame({'uid': [4, 5, 6], 'name': ['Dave', 'Eve', '']})
        pytest.raises(UniqueViolation, pg.df_copy_rows, df=df3, schema_nm=schema_nm, tb_nm=tb_nm)
        pg.drop_table(schema_nm, tb_nm)

    def test_df_insert_rows(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        pg.df_create_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        res = pg.df_insert_rows(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
        assert res == 3
        df2 = pd.DataFrame({'id': [3, 5, 6], 'name': ['Dave', 'Eve', 'Frank']})
        res = pg.df_insert_rows(df=df2, schema_nm=schema_nm, tb_nm=tb_nm, on_conflict='upsert')
        assert pg.select_rows(schema_nm, tb_nm) == (
            ['id', 'name'],
            [(1, 'Alice'), (2, 'Bob'), (3, 'Dave'), (5, 'Eve'), (6, 'Frank')],
        )
        pg.drop_table(schema_nm, tb_nm)

    def test_df_upsert_table(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'notes': [None, None, None]})
        pg.df_upsert_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        assert pg.select_rows(schema_nm, tb_nm) == (
            ['id', 'name', 'notes'],
            [(1, 'Alice', None), (2, 'Bob', None), (3, 'Charlie', None)],
        )
        assert isinstance(pg.get_column(schema_nm=schema_nm, tb_nm='test_table_2', col_nm='notes').type, String)
        df2 = pd.DataFrame({'id': [3, 5, 6], 'name': ['Dave', 'Eve', 'Frank'], 'notes': ['A', 'B', 'C']})
        pg.df_upsert_table(df=df2, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        assert pg.select_rows(schema_nm, tb_nm) == (
            ['id', 'name', 'notes'],
            [(1, 'Alice', None), (2, 'Bob', None), (3, 'Dave', 'A'), (5, 'Eve', 'B'), (6, 'Frank', 'C')],
        )
        df3 = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'notes': [['A', 'B'], ['B'], []]})
        pg.df_upsert_table(df=df3, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        assert pg.select_rows(schema_nm=schema_nm, tb_nm=tb_nm, order=Order('id')) == (
            ['id', 'name', 'notes'],
            [(1, 'Alice', ['A', 'B']), (2, 'Bob', ['B']), (3, 'Charlie', []), (5, 'Eve', 'B'), (6, 'Frank', 'C')],
        )
        pg.drop_table(schema_nm, tb_nm)

    def test_df_select_rows(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        pg.df_upsert_table(df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='id')
        res = pg.df_select_rows(schema_nm=schema_nm, tb_nm=tb_nm)
        assert res.equals(df)
        pg.drop_table(schema_nm, tb_nm)

    def test_df_primary_key(self, pg: PgsqlDF):
        schema_nm = 'test_schema'

        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [17, 18, 19]})
        df2 = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [18, 19, 20]})
        pg.df_create_table(
            df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm='index', unique_col_nms=['name'], notnull_col_nms=['name']
        )
        pg.df_insert_rows(df=df, schema_nm=schema_nm, tb_nm=tb_nm, on_conflict='update')
        pg.df_insert_rows(df=df2, schema_nm=schema_nm, tb_nm=tb_nm, on_conflict='update')
        pkey = pg.get_table(schema_nm=schema_nm, tb_nm=tb_nm).primary_key
        assert pkey.c[0].name == 'uid'
        pg.drop_table(schema_nm, tb_nm)

        tb_nm = 'Test_Table_3'
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [17, 18, 19]})
        df2 = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [18, 19, 20]})
        pg.df_create_table(
            df=df, schema_nm=schema_nm, tb_nm=tb_nm, primary_col_nm=None, unique_col_nms=['name'], notnull_col_nms=['name']
        )
        pg.df_insert_rows(df=df, schema_nm=schema_nm, tb_nm=tb_nm, on_conflict='update')
        pg.df_insert_rows(df=df2, schema_nm=schema_nm, tb_nm=tb_nm, on_conflict='update')
        pkey = pg.get_table(schema_nm=schema_nm, tb_nm=tb_nm).primary_key
        assert pkey.name is None
        pg.drop_table(schema_nm, tb_nm)

    def test_df_execute_select_sql(self, pg: PgsqlDF):
        schema_nm = 'test_schema'
        pg.create_table(
            schema_nm=schema_nm, tb_nm='test_table_2', cols=[Column('id', Integer, primary_key=True), Column('name', String)]
        )
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 1, 'name': 'Alice'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 2, 'name': 'Bob'})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_2', row={'id': 3, 'name': 'Charlie'})
        pg.create_table(
            schema_nm=schema_nm, tb_nm='test_table_3', cols=[Column('id', Integer, primary_key=True), Column('age', Integer)]
        )
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 1, 'age': 20})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 2, 'age': 30})
        pg.insert_row(schema_nm=schema_nm, tb_nm='test_table_3', row={'id': 3, 'age': 40})
        df = pg.df_execute_select(f'SELECT * FROM {schema_nm}.test_table_2')
        assert df.equals(pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}).convert_dtypes())
        df = pg.df_execute_select(
            f'SELECT t2.name, t3.age FROM {schema_nm}.test_table_2 t2 JOIN {schema_nm}.test_table_3 t3 ON t2.id = t3.id'
        )
        assert df.equals(pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [20, 30, 40]}).convert_dtypes())
        pg.drop_table(schema_nm, 'test_table_2')
        pg.drop_table(schema_nm, 'test_table_3')
