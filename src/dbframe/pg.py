import os
import pathlib
from typing import Any, Literal, Sequence
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    Column,
    Constraint,
    CursorResult,
    Index,
    MetaData,
    Row,
    Table,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.util import FacadeDict

from .clause import Order, Where, order_parser, where_parser
from .dtype import df_to_rows, df_to_schema_items
from .validator import NameValidator
from .sqlite import Sqlite


class Pg:
    def __init__(
        self,
        host: str = None,
        port: int | str = None,
        user: str = None,
        password: str = None,
        dbname: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.host = host or os.getenv('PG_HOST', 'localhost')
        self.port = port or os.getenv('PG_PORT', 5432)
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = password or os.getenv('PG_PASS', 'postgres')
        self.dbname = dbname or os.getenv('PG_DBNAME', 'postgres')
        self.dbname = NameValidator.dbname(self.dbname)
        self.verbose = verbose
        self.url = self.get_url()
        self.engine = create_engine(
            self.url,
            isolation_level='AUTOCOMMIT',
            executemany_mode='values_plus_batch',
            **kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __del__(self):
        self.engine.dispose()

    def _verbose_print(self, msg: Any):
        if self.verbose:
            print(msg)

    def _get_url(self, dbname: str) -> str:
        url_template = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'
        return url_template.format(
            user=self.user,
            password=quote_plus(self.password),
            host=self.host,
            port=self.port,
            dbname=dbname,
        )

    def get_url(self) -> str:
        return self._get_url(dbname=self.dbname)
    
    def validate_conn(self) -> bool:
        with self.engine.connect():
            return True

    # Database CRUD
    def create_database(self, dbname: str, **kwargs) -> str:
        dbname = NameValidator.dbname(dbname)
        engine = create_engine(self.url_default)
        with engine.connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchall()
            databases = [row[0] for row in rows]
            return databases
        ...

    def get_database(self, dbname: str) -> str | None:
        dbname = NameValidator.dbname(dbname)
        stmt = text(f'SELECT datname FROM pg_catalog.pg_database WHERE datname = {dbname};')
        engine = create_engine(self.url_default)
        with engine.connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchall()
            databases = [row[0] for row in rows]
            return databases
        ...

    def get_databases(self, dbname: str) -> str | None:
        ...

    def drop_database(self):
        ...

    # Table CRUD
    def get_table(self):
        ...

    def get_tables(self):
        ...

    def create_table(self):
        ...

    def rename_table(self):
        ...

    def drop_table(self):
        ...

    def truncate_table(self):
        ...

    # Column CRUD
    def add_column(self):
        ...

    def add_columns(self):
        ...

    def get_column(self):
        ...

    def get_columns(self):
        ...

    def rename_column(self):
        ...

    def alter_column(self):
        ...

    def drop_column(self):
        ...

    def drop_columns(self):
        ...

    # Index CRUD
    def create_index(self):
        ...

    def get_index(self):
        ...

    def get_indexes(self):
        ...

    def drop_index(self):
        ...

    # Row CRUD
    def insert_row(self):
        ...

    def insert_rows(self):
        ...

    def select_rows(
        self,
        schema_nm: str,
        tb_nm: str,
        col_nms: Sequence[str] = None,
        where: Where | Sequence = None,
        order: Order | Sequence[Order] = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ):
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nms = [NameValidator.column(col_nm) for col_nm in col_nms] if col_nms else None
        ...

    def update_rows(self):
        ...

    def delete_rows(self):
        ...

    # SQL Execution
    def _execute_sql(self, sql: str):
        ...


class PgDF(Pg):
    def __init__(self):
        super().__init__()
        pass

    def df_create_table(self):
        ...

    def df_add_columns(self):
        ...

    def df_alter_columns(self):
        ...

    def df_copy_rows(self):
        ...

    def df_insert_rows(self):
        ...

    def df_select_rows(self):
        ...
