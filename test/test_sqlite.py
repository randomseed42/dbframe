import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.exc import IntegrityError, OperationalError

from dbframe import Order, Sqlite, SqliteDF, Where


@pytest.fixture
def tmp_dir(tmp_path_factory):
    _tmp_dir = tmp_path_factory.mktemp('test_sqlite')
    yield _tmp_dir


class TestSqliteDatabase:
    def test_get_url(self, tmp_dir):
        cwd = os.path.abspath(os.getcwd())
        assert Sqlite().url == 'sqlite:///:memory:'
        assert Sqlite(db_path=':memory:').url == 'sqlite:///:memory:'
        for db_path in (
            'data.db',
            './data.db',
            '../data.db',
            '../../data.db',
            'test/data.db',
        ):
            url = Sqlite(db_path=db_path).url
            target_url = f'sqlite:///{os.path.abspath(os.path.normpath(os.path.join(cwd, db_path)))}'
            assert url == target_url
        for db_path in (
            os.path.join(tmp_dir, 'data.db'),
            os.path.join(tmp_dir, './data.db'),
            os.path.join(tmp_dir, '../data.db'),
            os.path.join(tmp_dir, '../../data.db'),
        ):
            url = Sqlite(db_path=db_path).url
            target_url = f'sqlite:///{os.path.abspath(os.path.normpath(db_path))}'
            assert url == target_url
        for db_path in (
            Path(tmp_dir, 'data.db'),
            Path(tmp_dir, './data.db'),
            Path(tmp_dir, '../data.db'),
            Path(tmp_dir, '../../data.db'),
        ):
            url = Sqlite(db_path=db_path).url
            target_url = f'sqlite:///{os.path.abspath(os.path.normpath(db_path))}'
            assert url == target_url

    def test_validate_conn(self, tmp_dir):
        assert Sqlite().validate_conn() is True
        assert Sqlite(db_path=Path(tmp_dir, 'data.db')).validate_conn() is True
        if os.name == 'nt':
            pytest.raises(OperationalError, Sqlite(db_path=r'c:\Windows\data.db', verbose=True).validate_conn)

    def test_create_database(self, tmp_dir):
        assert Sqlite().create_database(db_path=Path(tmp_dir, 'data.db')) == str(Path(tmp_dir, 'data.db'))
        assert Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database(db_path=Path(tmp_dir, 'data2.db')) == str(
            Path(tmp_dir, 'data2.db')
        )
        pytest.raises(ValueError, Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database, db_path=':memory:')
        pytest.raises(ValueError, Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database, db_path=':invalid_file:')
        pytest.raises(TypeError, Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database)

    def test_get_database(self, tmp_dir):
        Sqlite().create_database(db_path=Path(tmp_dir, 'data.db'))
        assert Sqlite().get_database(db_path=Path(tmp_dir, 'data.db')) == str(Path(tmp_dir, 'data.db'))
        assert Sqlite().get_database(db_path=Path(tmp_dir, 'data2.db')) is None

    def test_drop_database(self, tmp_dir):
        Sqlite().create_database(db_path=Path(tmp_dir, 'data.db'))
        assert Sqlite().drop_database(db_path=Path(tmp_dir, 'data.db')) == str(Path(tmp_dir, 'data.db'))
        assert Sqlite().drop_database(db_path=Path(tmp_dir, 'data2.db')) is None


class TestSqliteTable:
    def test_get_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        Sqlite().create_database(db_path=db_path)
        db = Sqlite(db_path=db_path)
        pytest.raises(ValueError, db.get_table, tb_nm='test_table')
        cols = [Column('id', Integer, primary_key=True), Column('name', String)]
        db.create_table(tb_nm='test_table', cols=cols)
        assert db.get_table(tb_nm='test_table').name == 'test_table'

    def test_get_tables(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        Sqlite().create_database(db_path=db_path)
        db = Sqlite(db_path=db_path)
        assert len(db.get_tables()) == 0
        cols1 = [Column('id', Integer, primary_key=True), Column('name', String)]
        cols2 = [Column('id', Integer, primary_key=True), Column('name', String)]
        db.create_table(tb_nm='test_table1', cols=cols1)
        db.create_table(tb_nm='test_table2', cols=cols2)
        assert tuple(db.get_tables().keys()) == ('test_table1', 'test_table2')

    def test_create_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        # Sqlite().create_database(db_path=db_path)
        db = Sqlite(db_path=db_path)
        cols = [Column('id', Integer, primary_key=True), Column('name', String)]
        db.create_table(tb_nm='test_table', cols=cols)
        assert db.get_table(tb_nm='test_table').name == 'test_table'
        cols = [Column('id', Integer, primary_key=True), Column('name', String)]
        pytest.raises(ValueError, db.create_table, tb_nm='test_table', cols=cols)
        assert db.get_table(tb_nm='test_table').columns.keys() == ['id', 'name']
        assert db.get_table(tb_nm='test_table').columns['id'].primary_key is True
        assert isinstance(db.get_table(tb_nm='test_table').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table').columns['name'].type.length is None
        cols = [Column('id', Integer, primary_key=True), Column('name', String(50))]
        db.create_table(tb_nm='test_table2', cols=cols)
        assert isinstance(db.get_table(tb_nm='test_table2').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table2').columns['name'].type.length == 50
        cols = [Column('id', Integer, primary_key=True), Column('name', String(50)), Column('age', Integer)]
        db.create_table(tb_nm='test_table3', cols=cols)
        assert db.get_table(tb_nm='test_table3').columns.keys() == ['id', 'name', 'age']
        assert isinstance(db.get_table(tb_nm='test_table3').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table3').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table3').columns['age'].type, Integer)
        cols = [Column('id', Integer, primary_key=True), Column('name', String(50)), Column('age', Integer, nullable=False)]
        db.create_table(tb_nm='test_table4', cols=cols)
        assert db.get_table(tb_nm='test_table4').columns.keys() == ['id', 'name', 'age']
        assert isinstance(db.get_table(tb_nm='test_table4').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table4').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table4').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table4').columns['age'].nullable is False
        cols = [Column('id', Integer, primary_key=True), Column('name', String(50)), Column('age', Integer, nullable=True)]
        db.create_table(tb_nm='test_table5', cols=cols)
        assert db.get_table(tb_nm='test_table5').columns.keys() == ['id', 'name', 'age']
        assert isinstance(db.get_table(tb_nm='test_table5').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table5').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table5').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table5').columns['age'].nullable is True
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer, nullable=True, default=0),
        ]
        db.create_table(tb_nm='test_table6', cols=cols)
        assert db.get_table(tb_nm='test_table6').columns.keys() == ['id', 'name', 'age']
        assert isinstance(db.get_table(tb_nm='test_table6').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table6').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table6').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table6').columns['age'].nullable is True
        assert db.get_table(tb_nm='test_table6').columns['age'].default is None
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer, nullable=True, default=0),
            Column('salary', Float),
        ]
        db.create_table(tb_nm='test_table7', cols=cols)
        assert db.get_table(tb_nm='test_table7').columns.keys() == ['id', 'name', 'age', 'salary']
        assert isinstance(db.get_table(tb_nm='test_table7').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table7').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table7').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table7').columns['age'].nullable is True
        assert db.get_table(tb_nm='test_table7').columns['age'].default is None
        assert isinstance(db.get_table(tb_nm='test_table7').columns['salary'].type, Float)
        assert db.get_table(tb_nm='test_table7').columns['salary'].nullable is True
        assert db.get_table(tb_nm='test_table7').columns['salary'].default is None
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer, nullable=True, default=0),
            Column('salary', Float, nullable=False, default=0.0),
        ]
        db.create_table(tb_nm='test_table8', cols=cols)
        assert db.get_table(tb_nm='test_table8').columns.keys() == ['id', 'name', 'age', 'salary']
        assert isinstance(db.get_table(tb_nm='test_table8').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table8').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table8').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table8').columns['age'].nullable is True
        assert db.get_table(tb_nm='test_table8').columns['age'].default is None
        assert isinstance(db.get_table(tb_nm='test_table8').columns['salary'].type, Float)
        assert db.get_table(tb_nm='test_table8').columns['salary'].nullable is False
        assert db.get_table(tb_nm='test_table8').columns['salary'].default is None
        cols = [
            Column('id', Integer, primary_key=True),
            Column('name', String(50)),
            Column('age', Integer, nullable=True, default=0),
            Column('salary', Float, nullable=False, default=0.0),
            Column('is_active', Boolean),
        ]
        db.create_table(tb_nm='test_table9', cols=cols)
        assert db.get_table(tb_nm='test_table9').columns.keys() == ['id', 'name', 'age', 'salary', 'is_active']
        assert isinstance(db.get_table(tb_nm='test_table9').columns['name'].type, String)
        assert db.get_table(tb_nm='test_table9').columns['name'].type.length == 50
        assert isinstance(db.get_table(tb_nm='test_table9').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table9').columns['age'].nullable is True
        assert db.get_table(tb_nm='test_table9').columns['age'].default is None
        assert isinstance(db.get_table(tb_nm='test_table9').columns['salary'].type, Float)
        assert db.get_table(tb_nm='test_table9').columns['salary'].nullable is False
        assert db.get_table(tb_nm='test_table9').columns['salary'].default is None
        assert isinstance(db.get_table(tb_nm='test_table9').columns['is_active'].type, Boolean)
        assert db.get_table(tb_nm='test_table9').columns['is_active'].nullable is True
        assert db.get_table(tb_nm='test_table9').columns['is_active'].default is None

    def test_rename_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.rename_table(tb_nm='test_table', new_tb_nm='test_table2')
        assert db.get_table(tb_nm='test_table2').name == 'test_table2'
        pytest.raises(ValueError, db.get_table, tb_nm='test_table')

    def test_drop_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.drop_table(tb_nm='test_table')
        pytest.raises(ValueError, db.get_table, tb_nm='test_table')
        pytest.raises(ValueError, db.drop_table, tb_nm='test_table')

    def test_truncate_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(
            tb_nm='test_table',
            cols=[Column('id', Integer, primary_key=True, autoincrement=True), Column('name', String)],
            sqlite_autoincrement=True,
        )
        db.insert_row(tb_nm='test_table', row={'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'name': 'Charlie'})
        db.truncate_table(tb_nm='test_table', restart=False)
        assert len(db.select_rows(tb_nm='test_table')[1]) == 0
        db.insert_row(tb_nm='test_table', row={'name': 'David'})
        assert db.select_rows(tb_nm='test_table')[1] == [(4, 'David')]
        db.truncate_table(tb_nm='test_table', restart=True)
        db.insert_row(tb_nm='test_table', row={'name': 'Eddy'})
        assert db.select_rows(tb_nm='test_table')[1] == [(1, 'Eddy')]

        db_path = Path(tmp_dir, 'data2.db')
        db = Sqlite(db_path=db_path)
        db.create_table(
            tb_nm='test_table',
            cols=[Column('id', Integer, primary_key=True, autoincrement=False), Column('name', String)],
            sqlite_autoincrement=False,
        )
        db.insert_row(tb_nm='test_table', row={'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'name': 'Charlie'})
        db.truncate_table(tb_nm='test_table', restart=False)
        assert len(db.select_rows(tb_nm='test_table')[1]) == 0
        db.insert_row(tb_nm='test_table', row={'name': 'David'})
        assert db.select_rows(tb_nm='test_table')[1] == [(1, 'David')]
        db.truncate_table(tb_nm='test_table', restart=True)
        db.insert_row(tb_nm='test_table', row={'name': 'Eddy'})
        assert db.select_rows(tb_nm='test_table')[1] == [(1, 'Eddy')]


class TestSqliteColumn:
    def test_add_column(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.add_column(tb_nm='test_table', col=Column('age', Integer))
        assert db.get_table(tb_nm='test_table').columns.keys() == ['id', 'name', 'age']
        assert isinstance(db.get_table(tb_nm='test_table').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table').columns['age'].nullable is True
        pytest.raises(OperationalError, db.add_column, tb_nm='test_table', col=Column('age', Integer))
        pytest.raises(OperationalError, db.add_column, tb_nm='test_table_2', col=Column('age', Integer))

    def test_add_columns(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.add_columns(tb_nm='test_table', cols=[Column('age', Integer), Column('salary', Float)])
        assert db.get_table(tb_nm='test_table').columns.keys() == ['id', 'name', 'age', 'salary']
        assert isinstance(db.get_table(tb_nm='test_table').columns['age'].type, Integer)
        assert db.get_table(tb_nm='test_table').columns['age'].nullable is True
        assert isinstance(db.get_table(tb_nm='test_table').columns['salary'].type, Float)
        assert db.get_table(tb_nm='test_table').columns['salary'].nullable is True

    def test_get_column(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        assert db.get_column(tb_nm='test_table', col_nm='id').name == 'id'
        assert db.get_column(tb_nm='test_table', col_nm='name').name == 'name'
        pytest.raises(ValueError, db.get_column, tb_nm='test_table', col_nm='age')

    def test_get_columns(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        assert list(db.get_columns(tb_nm='test_table').keys()) == ['id', 'name']

    def test_alter_column(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.alter_column(tb_nm='test_table', col_nm='name', new_col_nm='full_name')
        assert list(db.get_table(tb_nm='test_table').columns.keys()) == ['id', 'full_name']
        assert db.get_column(tb_nm='test_table', col_nm='full_name').name == 'full_name'
        pytest.raises(OperationalError, db.rename_column, tb_nm='test_table', col_nm='full_name', new_col_nm='id')
        pytest.raises(
            NotImplementedError, db.alter_column, tb_nm='test_table', col_nm='full_name', new_col_nm='name', sql_dtype='str'
        )

    def test_drop_column(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.drop_column(tb_nm='test_table', col_nm='name')
        assert list(db.get_table(tb_nm='test_table').columns.keys()) == ['id']
        pytest.raises(ValueError, db.get_column, tb_nm='test_table', col_nm='name')

    def test_drop_columns(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(
            tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String), Column('age', Integer)]
        )
        db.drop_columns(tb_nm='test_table', col_nms=['name', 'age'])
        assert list(db.get_table(tb_nm='test_table').columns.keys()) == ['id']
        pytest.raises(ValueError, db.get_column, tb_nm='test_table', col_nm='name')
        pytest.raises(ValueError, db.get_column, tb_nm='test_table', col_nm='age')


class TestSqliteIndex:
    def test_create_index(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.create_index(tb_nm='test_table', col_nms=['name'])
        assert 'idx_test_table_name' in db.get_indexes(tb_nm='test_table')
        assert list(db.get_indexes(tb_nm='test_table')['idx_test_table_name'].columns.keys()) == ['name']
        assert db.get_indexes(tb_nm='test_table')['idx_test_table_name'].unique == 0
        db.create_index(tb_nm='test_table', col_nms=['id', 'name'], idx_nm='idx_test_table_id_name', unique=True)
        assert 'idx_test_table_id_name' in db.get_indexes(tb_nm='test_table')
        assert list(db.get_indexes(tb_nm='test_table')['idx_test_table_id_name'].columns.keys()) == ['id', 'name']
        assert db.get_indexes(tb_nm='test_table')['idx_test_table_id_name'].unique == 1
        db.create_table(tb_nm='test_table2', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        pytest.raises(
            OperationalError,
            db.create_index,
            tb_nm='test_table2',
            col_nms=['id', 'name'],
            idx_nm='idx_test_table_id_name',
            unique=True,
        )

    def test_get_index(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.create_index(tb_nm='test_table', col_nms=['name'])
        assert db.get_index(tb_nm='test_table', idx_nm='idx_test_table_name').name == 'idx_test_table_name'
        assert list(db.get_index(tb_nm='test_table', idx_nm='idx_test_table_name').columns.keys()) == ['name']
        assert db.get_index(tb_nm='test_table', idx_nm='idx_test_table_name').unique == 0
        db.create_index(tb_nm='test_table', col_nms=['id', 'name'], idx_nm='idx_test_table_id_name', unique=True)
        assert db.get_index(tb_nm='test_table', idx_nm='idx_test_table_id_name').name == 'idx_test_table_id_name'
        assert list(db.get_index(tb_nm='test_table', idx_nm='idx_test_table_id_name').columns.keys()) == ['id', 'name']
        assert db.get_index(tb_nm='test_table', idx_nm='idx_test_table_id_name').unique == 1
        pytest.raises(ValueError, db.get_index, tb_nm='test_table', idx_nm='idx_test_table_age')

    def test_get_indexes(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.create_index(tb_nm='test_table', col_nms=['name'])
        assert 'idx_test_table_name' in db.get_indexes(tb_nm='test_table')
        assert list(db.get_indexes(tb_nm='test_table')['idx_test_table_name'].columns.keys()) == ['name']
        assert db.get_indexes(tb_nm='test_table')['idx_test_table_name'].unique == 0
        db.create_index(tb_nm='test_table', col_nms=['id', 'name'], idx_nm='idx_test_table_id_name', unique=True)
        assert 'idx_test_table_id_name' in db.get_indexes(tb_nm='test_table')
        assert list(db.get_indexes(tb_nm='test_table')['idx_test_table_id_name'].columns.keys()) == ['id', 'name']
        assert db.get_indexes(tb_nm='test_table')['idx_test_table_id_name'].unique == 1
        db.create_table(tb_nm='test_table2', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        assert db.get_indexes(tb_nm='test_table2') == {}

    def test_drop_index(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.create_index(tb_nm='test_table', col_nms=['name'])
        db.drop_index(tb_nm='test_table', idx_nm='idx_test_table_name')
        assert 'idx_test_table_name' not in db.get_indexes(tb_nm='test_table')
        db.create_index(tb_nm='test_table', col_nms=['id', 'name'], idx_nm='idx_test_table_id_name', unique=True)
        db.drop_index(tb_nm='test_table', idx_nm='idx_test_table_id_name')
        assert 'idx_test_table_id_name' not in db.get_indexes(tb_nm='test_table')
        db.create_table(tb_nm='test_table2', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        pytest.raises(ValueError, db.drop_index, tb_nm='test_table2', idx_nm='idx_test_table_id_name')


class TestSqliteRow:
    def test_insert_row(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        assert db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'}) == 1
        assert db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'}) == 1
        assert db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'}) == 1
        pytest.raises(IntegrityError, db.insert_row, tb_nm='test_table', row={'id': 3, 'name': 'Carl'})
        assert db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Carl'}, on_conflict='do_nothing') == 0
        assert db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Carl'}, on_conflict='upsert') == 1
        assert db.insert_row(tb_nm='test_table', row={'name': 'Dave'}) == 1
        assert db.insert_row(tb_nm='test_table', row={'id': 6, 'name': 'Foxy'}) == 1
        assert db.insert_row(tb_nm='test_table', row={'NAME': 'Eddy'}) == 1
        assert db.select_rows(tb_nm='test_table') == (
            ['id', 'name'],
            [(1, 'Alice'), (2, 'Bob'), (3, 'Carl'), (4, 'Dave'), (6, 'Foxy'), (7, None)],
        )
        assert (
            db.insert_row(tb_nm='test_table', row={'Id': 7, 'NAME': 'Eddy'}, on_conflict='upsert', row_key_validate=True) == 1
        )
        assert db.select_rows(tb_nm='test_table') == (
            ['id', 'name'],
            [(1, 'Alice'), (2, 'Bob'), (3, 'Carl'), (4, 'Dave'), (6, 'Foxy'), (7, 'Eddy')],
        )

    def test_insert_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        assert (
            db.insert_rows(
                tb_nm='test_table', rows=[{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}, {'id': 3, 'name': 'Charlie'}]
            )
            == 3
        )
        pytest.raises(
            IntegrityError, db.insert_rows, tb_nm='test_table', rows=[{'id': 3, 'name': 'Carl'}, {'id': 4, 'name': 'Dave'}]
        )
        assert (
            db.insert_rows(
                tb_nm='test_table', rows=[{'id': 3, 'name': 'Carl'}, {'id': 4, 'name': 'Dave'}], on_conflict='do_nothing'
            )
            == 1
        )
        assert (
            db.insert_rows(
                tb_nm='test_table', rows=[{'id': 3, 'name': 'Carl'}, {'id': 4, 'name': 'Dave'}], on_conflict='ignore'
            )
            == 0
        )
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie'), (4, 'Dave')])
        res = db.insert_rows(
            tb_nm='test_table', rows=[{'id': 3, 'name': 'Carl'}, {'id': 4, 'name': 'Dave'}], on_conflict='upsert'
        )
        assert res == 2
        res = db.select_rows(tb_nm='test_table')
        assert res == (['id', 'name'], [(1, 'Alice'), (2, 'Bob'), (3, 'Carl'), (4, 'Dave')])

    def test_select_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'})
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')])
        assert db.select_rows(tb_nm='test_table', col_nms=['name']) == (['name'], [('Alice',), ('Bob',), ('Charlie',)])
        pytest.raises(ValueError, db.select_rows, tb_nm='test_table', col_nms=[])
        pytest.raises(ValueError, db.select_rows, tb_nm='test_table', col_nms=['name', 'age'])
        assert db.select_rows(tb_nm='test_table', col_nms=['name'], where=Where('id', '>', 1), order=Order('id', False)) == (
            ['name'],
            [('Charlie',), ('Bob',)],
        )
        assert db.select_rows(
            tb_nm='test_table',
            col_nms=['name'],
            where=[Where('id', '>', 1), Where('name', 'like', 'B%')],
            order=Order('id', False),
        ) == (['name'], [('Bob',)])
        assert db.select_rows(
            tb_nm='test_table',
            col_nms=['name'],
            where=(Where('id', '>', 2), Where('name', 'like', 'A%')),
            order=Order('id', False),
        ) == (['name'], [('Charlie',), ('Alice',)])
        assert db.select_rows(
            tb_nm='test_table',
            col_nms=['name'],
            where=(Where('id', '<', 2), [Where('id', '>=', 2), Where('name', 'ilike', '%li%')]),
            order=Order('id', False),
        ) == (['name'], [('Charlie',), ('Alice',)])

    def test_update_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'})
        assert db.update_rows(tb_nm='test_table', set_values={'name': 'Eve'}, where=Where('id', '>', 1)) == 2
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [(1, 'Alice'), (2, 'Eve'), (3, 'Eve')])
        assert db.update_rows(tb_nm='test_table', set_values={'name': 'Foxy'}, where=Where('id', '<', 3)) == 2
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [(1, 'Foxy'), (2, 'Foxy'), (3, 'Eve')])

    def test_delete_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'})
        assert db.delete_rows(tb_nm='test_table', where=Where('id', '>', 1)) == 2
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [(1, 'Alice')])
        assert db.delete_rows(tb_nm='test_table') == 1
        assert db.select_rows(tb_nm='test_table') == (['id', 'name'], [])

    def test_execute_sql(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = Sqlite(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'})
        db.create_table(tb_nm='test_table2', cols=[Column('id', Integer, primary_key=True), Column('age', Integer)])
        db.insert_row(tb_nm='test_table2', row={'id': 1, 'age': 20})
        db.insert_row(tb_nm='test_table2', row={'id': 2, 'age': 30})
        db.insert_row(tb_nm='test_table2', row={'id': 3, 'age': 40})
        assert db._execute_sql('SELECT * FROM test_table').fetchall() == [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
        assert db._execute_sql('SELECT t1.name, t2.age FROM test_table t1 JOIN test_table2 t2 ON t1.id = t2.id').fetchall() == [
            ('Alice', 20),
            ('Bob', 30),
            ('Charlie', 40),
        ]


class TestSqliteDF:
    def test_df_create_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        db.df_create_table(df=df, tb_nm='test_table', primary_col_nm='id')
        assert list(db.get_tables().keys()) == ['test_table']
        assert list(db.get_columns(tb_nm='test_table').keys()) == ['id', 'name']

    def test_df_insert_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [20, None, 40],
                'gender': ['F', 'M', ''],
                'BDay': [None, '2010-01-01', '2011-05-01'],
                'data': [b'a', b'b', b'\xff'],
            }
        )
        df['BDay'] = pd.to_datetime(df['BDay'])
        df['data'] = df['data'].astype(bytes)
        db.df_create_table(df=df, tb_nm='test_table', primary_col_nm='id')
        db.df_insert_rows(df=df, tb_nm='test_table')
        assert db.select_rows(tb_nm='test_table') == (
            ['id', 'name', 'age', 'gender', 'bday', 'data'],
            [
                (1, 'Alice', 20, 'F', None, b'a'),
                (2, 'Bob', None, 'M', '2010-01-01 00:00:00', b'b'),
                (3, 'Charlie', 40, None, '2011-05-01 00:00:00', b'\xff'),
            ],
        )

    def test_df_upsert_table(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'notes': [['A', 'B'], ['B'], []]})
        db.df_upsert_table(df=df, tb_nm='test_table', primary_col_nm='id')
        assert db.select_rows(tb_nm='test_table', order=Order('id')) == (
            ['id', 'name', 'notes'],
            [(1, 'Alice', '["A","B"]'), (2, 'Bob', '["B"]'), (3, 'Charlie', '[]')],
        )

    def test_df_select_rows(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        db.df_create_table(df=df, tb_nm='test_table', primary_col_nm='id')
        db.df_insert_rows(df=df, tb_nm='test_table')
        pd.testing.assert_frame_equal(db.df_select_rows(tb_nm='test_table'), df)
        pd.testing.assert_frame_equal(
            db.df_select_rows(tb_nm='test_table', col_nms=['name']),
            pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']}).convert_dtypes(),
        )
        pytest.raises(ValueError, db.df_select_rows, tb_nm='test_table', col_nms=[])
        pytest.raises(ValueError, db.df_select_rows, tb_nm='test_table', col_nms=['name', 'age'])

    def test_df_primary_key(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)

        tb_nm = 'Test_Table_2'
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [17, 18, 19]})
        df2 = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [18, 19, 20]})
        db.df_create_table(df=df, tb_nm=tb_nm, primary_col_nm='index', unique_col_nms=['name'], notnull_col_nms=['name'])
        db.df_insert_rows(df=df, tb_nm=tb_nm, on_conflict='update')
        db.df_insert_rows(df=df2, tb_nm=tb_nm, on_conflict='update')
        pkey = db.get_table(tb_nm=tb_nm).primary_key
        assert pkey.c[0].name == 'uid'
        db.drop_table(tb_nm)

        tb_nm = 'Test_Table_3'
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [17, 18, 19]})
        df2 = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [18, 19, 20]})
        db.df_create_table(df=df, tb_nm=tb_nm, primary_col_nm=None, unique_col_nms=['name'], notnull_col_nms=['name'])
        db.df_insert_rows(df=df, tb_nm=tb_nm, on_conflict='update')
        db.df_insert_rows(df=df2, tb_nm=tb_nm, on_conflict='update')
        pkey = db.get_table(tb_nm=tb_nm).primary_key
        assert pkey.name is None
        db.drop_table(tb_nm)

    def test_df_execute_select(self, tmp_dir):
        db_path = Path(tmp_dir, 'data.db')
        db = SqliteDF(db_path=db_path)
        db.create_table(tb_nm='test_table', cols=[Column('id', Integer, primary_key=True), Column('name', String)])
        db.insert_row(tb_nm='test_table', row={'id': 1, 'name': 'Alice'})
        db.insert_row(tb_nm='test_table', row={'id': 2, 'name': 'Bob'})
        db.insert_row(tb_nm='test_table', row={'id': 3, 'name': 'Charlie'})
        db.create_table(tb_nm='test_table2', cols=[Column('id', Integer, primary_key=True), Column('age', Integer)])
        db.insert_row(tb_nm='test_table2', row={'id': 1, 'age': 20})
        db.insert_row(tb_nm='test_table2', row={'id': 2, 'age': 30})
        db.insert_row(tb_nm='test_table2', row={'id': 3, 'age': 40})
        df = db.df_execute_select('SELECT * FROM test_table')
        assert df.equals(pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}).convert_dtypes())
        df = db.df_execute_select('SELECT t1.name, t2.age FROM test_table t1 JOIN test_table2 t2 ON t1.id = t2.id')
        assert df.equals(pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [20, 30, 40]}).convert_dtypes())
