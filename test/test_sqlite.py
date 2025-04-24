import os
from pathlib import Path

import pytest
from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.exc import OperationalError

from dbframe.sqlite import Sqlite


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
        pytest.raises(OperationalError, Sqlite(db_path=r'c:\Windows\data.db', verbose=True).validate_conn)

    def test_create_database(self, tmp_dir):
        assert Sqlite().create_database(db_path=Path(tmp_dir, 'data.db')) == str(Path(tmp_dir, 'data.db'))
        assert Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database(db_path=Path(tmp_dir, 'data2.db')) == str(
            Path(tmp_dir, 'data2.db')
        )
        pytest.raises(ValueError, Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database, db_path=':memory:')
        pytest.raises(OperationalError, Sqlite(db_path=Path(tmp_dir, 'data.db')).create_database, db_path=':invalid_file:')
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
        assert db.get_table(tb_nm='test_table') is None
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
        Sqlite().create_database(db_path=db_path)
        db = Sqlite(db_path=db_path)
        cols = [Column('id', Integer, primary_key=True), Column('name', String)]
        db.create_table(tb_nm='test_table', cols=cols)
        assert db.get_table(tb_nm='test_table').name == 'test_table'
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
