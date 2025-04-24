import os
import pathlib
from typing import Any

from sqlalchemy import Column, MetaData, Table, create_engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.util import FacadeDict

from .validator import NameValidator


class Sqlite:
    def __init__(self, db_path: str | os.PathLike | pathlib.Path = None, verbose: bool = False, **kwargs):
        self.db_path = db_path or ':memory:'
        self.verbose = verbose
        self.abs_db_path = self.get_abs_db_path(db_path=self.db_path)
        self.url = self.get_url()
        self.engine = create_engine(self.url, isolation_level='AUTOCOMMIT', connect_args={'check_same_thread': False})

    def __del__(self) -> None:
        self.engine.dispose()

    def _verbose_print(self, msg: Any):
        if self.verbose:
            print(msg)

    @staticmethod
    def get_abs_db_path(db_path: str | os.PathLike | pathlib.Path) -> str:
        if db_path == ':memory:':
            return db_path
        db_path = os.path.abspath(os.path.normpath(db_path))
        return db_path

    def get_url(self) -> str:
        return f'sqlite:///{self.abs_db_path}'

    def validate_conn(self) -> bool:
        with self.engine.connect():
            return True

    # Database CRUD
    def create_database(self, db_path: str | os.PathLike | pathlib.Path, **kwargs) -> str:
        if db_path == ':memory:':
            raise ValueError('No need to create a database in memory, just initiate a Sqlite instance.')
        abs_db_path = self.get_abs_db_path(db_path=db_path)
        if os.path.exists(abs_db_path) and os.path.isfile(abs_db_path):
            raise FileExistsError(f'Database already exists at {abs_db_path}')
        Sqlite(db_path=abs_db_path, **kwargs).validate_conn()
        self._verbose_print(f'Created database at {abs_db_path}')
        return abs_db_path

    def get_database(self, db_path: str | os.PathLike | pathlib.Path, **kwargs) -> str | None:
        abs_db_path = self.get_abs_db_path(db_path=db_path)
        if not os.path.exists(abs_db_path) or not os.path.isfile(abs_db_path):
            return
        if not Sqlite(db_path=abs_db_path, **kwargs).validate_conn():
            return
        return abs_db_path

    def drop_database(self, db_path: str | os.PathLike | pathlib.Path, **kwargs) -> str | None:
        abs_db_path = self.get_abs_db_path(db_path=db_path)
        if not os.path.exists(abs_db_path) or not os.path.isfile(abs_db_path):
            return
        if not Sqlite(db_path=abs_db_path, **kwargs).validate_conn():
            return
        os.remove(abs_db_path)
        self._verbose_print(f'Dropped database at {abs_db_path}')
        return abs_db_path

    # Table CRUD
    def get_table(self, tb_nm: str, **kwargs) -> Table | None:
        tb_nm = NameValidator.table(tb_nm)
        metadata = MetaData()
        try:
            tb = Table(tb_nm, metadata, autoload_with=self.engine, **kwargs)
            return tb
        except NoSuchTableError:
            self._verbose_print(f'Table {tb_nm} not found')
            return

    def get_tables(self, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        return metadata.tables

    def create_table(self, tb_nm: str, cols: list[Column], **kwargs) -> Table | None:
        tb_nm = NameValidator.table(tb_nm)
        if self.get_table(tb_nm=tb_nm) is not None:
            raise ValueError(f'Table {tb_nm} already exists')
        if len(cols) == 0:
            raise ValueError(f'Columns {cols} cannot be empty')
        metadata = MetaData()
        for col in cols:
            col.name = NameValidator.column(col.name)
        tb = Table(tb_nm, metadata, *cols, **kwargs)
        tb.create(bind=self.engine, checkfirst=True)
        self._verbose_print(f'Table {tb_nm} created')
        return self.get_table(tb_nm=tb_nm, **kwargs)


class SqliteDF(Sqlite):
    def __init__(self):
        super().__init__()
        pass
