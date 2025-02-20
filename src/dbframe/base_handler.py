from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd
from sqlalchemy import Column, Row, Table
from sqlalchemy.util import FacadeDict

from .logger import Logger
from .utils import OrderByClause, WhereClause


class BaseHandler(ABC):
    def __init__(
            self,
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        self.logger = Logger(
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        ).get_logger()
        self.logger.debug(f'logger initialized {self.logger.name=} {id(self.logger)=}')

    @abstractmethod
    def _generate_url(self, db_path: str, dbname: str, **kwargs) -> str:
        ...

    @abstractmethod
    def _validate_connection(self):
        ...

    # Database CRUD
    @abstractmethod
    def create_database(self, db_path: str, dbname: str, **kwargs) -> str | None:
        ...

    @abstractmethod
    def get_database(self, db_path: str, dbname: str, **kwargs) -> str | None:
        ...

    def get_databases(self) -> list[str]:
        ...

    @abstractmethod
    def drop_database(self, dbname: str, db_path: str, **kwargs) -> str | None:
        ...

    # Schema CRUD
    def create_schema(self, schema: str) -> str | None:
        ...

    def get_schema(self, schema: str) -> str | None:
        ...

    def get_schemas(self) -> list[str]:
        ...

    def drop_schema(self, schema: str, cascade: bool = False) -> str | None:
        ...

    # Table CRUD
    @abstractmethod
    def create_table(self, schema: str, table_name: str, columns: list[Column], **kwargs) -> Table | None:
        ...

    @abstractmethod
    def get_table(self, schema: str, table_name: str, **kwargs) -> Table | None:
        ...

    @abstractmethod
    def get_tables(self, schema: str, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        ...

    @abstractmethod
    def rename_table(self, schema: str, old_table_name: str, new_table_name: str, **kwargs) -> str | None:
        ...

    @abstractmethod
    def drop_table(self, schema: str, table_name: str, **kwargs) -> str | None:
        ...

    # Column CRUD
    @abstractmethod
    def add_column(self, schema: str, table_name: str, column: Column, **kwargs) -> str | None:
        ...

    @abstractmethod
    def add_columns(self, schema: str, table_name: str, columns: list[Column], **kwargs) -> list[str] | None:
        ...

    @abstractmethod
    def get_column(self, schema: str, table_name: str, column_name: str, **kwargs) -> Column | None:
        ...

    @abstractmethod
    def get_columns(self, schema: str, table_name: str, **kwargs) -> dict[str, Column] | None:
        ...

    @abstractmethod
    def alter_column(self, schema: str, table_name: str, column: str, new_column_name: str, **kwargs) -> str | None:
        ...

    @abstractmethod
    def drop_column(self, schema: str, table_name: str, column_name: str, **kwargs) -> str | None:
        ...

    @abstractmethod
    def drop_columns(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> list[str] | None:
        ...

    # Index CRUD
    @abstractmethod
    def create_index(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> str | None:
        ...

    @abstractmethod
    def get_indexes(self, schema: str, table_name: str, **kwargs) -> dict[str, list[str]] | None:
        ...

    @abstractmethod
    def drop_index(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> str | None:
        ...

    # Rows CRUD
    @abstractmethod
    def insert_rows(self, schema: str, table_name: str, rows: list[dict], on_conflict: Literal['do_nothing'] = None,
                    **kwargs) -> int | None:
        ...

    @abstractmethod
    def select_rows(self, schema: str, table_name: str, column_names: list[str] = None,
                    where_clauses: list | tuple | WhereClause = None, order_by: list[OrderByClause] = None,
                    offset: int = None, limit: int = None, **kwargs) -> tuple[list, list[Row]] | tuple[None, None]:
        ...

    @abstractmethod
    def update_rows(self, schema: str, table_name: str, set_clauses: dict[str, Any],
                    where_clauses: list | tuple | WhereClause = None, **kwargs) -> int | None:
        ...

    @abstractmethod
    def delete_rows(self, schema: str, table_name: str, where_clauses: list | tuple | WhereClause = None,
                    **kwargs) -> int | None:
        ...

    # Execute SQL
    @abstractmethod
    def _execute_sql(self, sql: str, return_result: bool = False) -> tuple[list[str], list[Row]] | None:
        ...


class BaseDFHandler(ABC):
    @abstractmethod
    def df_create_table(
            self,
            df: pd.DataFrame,
            schema: str,
            table_name: str,
            primary_column_name: str | Literal['index'] = None,
            primary_sql_column_name: str = None,
            notnull_column_names: list[str] = None,
            index_column_names: list[str | list[str]] = None,
            unique_column_names: list[str | list[str]] = None,
            insert_rows: bool = True,
            convert_df: bool = True,
            **kwargs
    ):
        ...

    @abstractmethod
    def df_add_columns(
            self,
            df: pd.DataFrame,
            schema: str,
            table_name: str,
            **kwargs
    ):
        ...

    @abstractmethod
    def df_alter_columns_type(
            self,
            df: pd.DataFrame,
            schema: str,
            table_name: str,
            convert_df: bool = True,
            **kwargs
    ):
        ...

    @abstractmethod
    def df_insert_rows(
            self,
            df: pd.DataFrame,
            schema: str,
            table_name: str,
            add_columns: bool = True,
            alter_columns_dtype: bool = True,
            on_conflict: Literal['do_nothing'] = None,
            convert_df: bool = True,
            **kwargs
    ):
        ...
