from typing import Literal, Any

from sqlalchemy import Table, Column, Row
from sqlalchemy.util import FacadeDict

from .logger import Logger
from .utils import WhereClause, OrderByClause


class BaseHandler:
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

    def _generate_url(self, db_path: str, dbname: str, **kwargs) -> str:
        ...

    def _validate_connection(self):
        ...

    # Database CRUD
    def create_database(self, db_path: str, dbname: str, **kwargs) -> str | None:
        ...

    def get_database(self, db_path: str, dbname: str, **kwargs) -> str | None:
        ...

    def get_databases(self) -> list[str]:
        ...

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
    def create_table(self, schema: str, table_name: str, columns: list[Column], **kwargs) -> Table | None:
        ...

    def get_table(self, schema: str, table_name: str, **kwargs) -> Table | None:
        ...

    def get_tables(self, schema: str, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        ...

    def rename_table(self, schema: str, old_table_name: str, new_table_name: str, **kwargs) -> str | None:
        ...

    def drop_table(self, schema: str, table_name: str, **kwargs) -> str | None:
        ...

    # Column CRUD
    def add_column(self, schema: str, table_name: str, column: Column, **kwargs) -> str | None:
        ...

    def add_columns(self, schema: str, table_name: str, columns: list[Column], **kwargs) -> list[str] | None:
        ...

    def get_column(self, schema: str, table_name: str, column_name: str, **kwargs) -> Column | None:
        ...

    def get_columns(self, schema: str, table_name: str, **kwargs) -> dict[str, Column] | None:
        ...

    def alter_column(self, schema: str, table_name: str, column: str, new_column_name: str, **kwargs) -> str | None:
        ...

    def drop_column(self, schema: str, table_name: str, column_name: str, **kwargs) -> str | None:
        ...

    def drop_columns(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> list[str] | None:
        ...

    # Index CRUD
    def create_index(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> str | None:
        ...

    def get_indexes(self, schema: str, table_name: str, **kwargs) -> dict[str, list[str]] | None:
        ...

    def drop_index(self, schema: str, table_name: str, column_names: list[str], **kwargs) -> str | None:
        ...

    # Rows CRUD
    def insert_rows(self, schema: str, table_name: str, rows: list[dict], on_conflict: Literal['do_nothing'] = None,
                    **kwargs) -> int | None:
        ...

    def select_rows(self, schema: str, table_name: str, column_names: list[str] = None,
                    where_clauses: list | tuple | WhereClause = None, order_by: list[OrderByClause] = None,
                    offset: int = None, limit: int = None, **kwargs) -> tuple[list, list[Row]] | tuple[None, None]:
        ...

    def update_rows(self, schema: str, table_name: str, set_clauses: dict[str, Any],
                    where_clauses: list | tuple | WhereClause = None, **kwargs) -> int | None:
        ...

    def delete_rows(self, schema: str, table_name: str, where_clauses: list | tuple | WhereClause = None,
                    **kwargs) -> int | None:
        ...

    # Execute SQL
    def _execute_sql(self, sql: str, return_result: bool = False) -> tuple[list[str], list[Row]] | None:
        ...
