from typing import Literal, Any

from sqlalchemy import Table, Column, Row

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
    def create_database(self, db_path: str, dbname: str, **kwargs):
        ...

    def get_database(self, db_path: str, dbname: str, **kwargs) -> str | None:
        ...

    def get_databases(self) -> list[str]:
        ...

    def drop_database(self, dbname: str, db_path: str, **kwargs) -> str:
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
    def create_table(self, table_name: str, schema: str, columns: list[Column], **kwargs) -> Table | None:
        ...

    def get_table(self, table_name: str, schema: str, **kwargs) -> Table | None:
        ...

    def get_tables(self, **kwargs) -> dict[str, Table]:
        ...

    def rename_table(self, old_table_name: str, new_table_name: str, schema: str, **kwargs) -> str | None:
        ...

    def drop_table(self, table_name: str, schema: str, **kwargs) -> str | None:
        ...

    # Column CRUD
    def add_column(self, table_name: str, schema: str, column: Column, **kwargs) -> str | None:
        ...

    def add_columns(self, table_name: str, schema: str, columns: list[Column], **kwargs) -> list[str] | None:
        ...

    def get_column(self, table_name: str, schema: str, column_name: str, **kwargs) -> Column | None:
        ...

    def get_columns(self, table_name: str, schema: str, **kwargs) -> dict[str, Column] | None:
        ...

    def alter_column(self, table_name: str, schema: str, column: str, new_column_name: str, **kwargs) -> str | None:
        ...

    def drop_column(self, table_name: str, schema: str, column_name: str, **kwargs) -> str | None:
        ...

    def drop_columns(self, table_name: str, schema: str, column_names: list[str], **kwargs) -> list[str] | None:
        ...

    # Index CRUD
    def create_index(self, table_name: str, schema: str, column_names: list[str], **kwargs) -> str | None:
        ...

    def get_indexes(self, table_name: str, schema: str, **kwargs) -> dict[str, list[str]] | None:
        ...

    def drop_index(self, table_name: str, schema: str, column_names: list[str], **kwargs) -> str | None:
        ...

    # Rows CRUD
    def insert_rows(self, table_name: str, schema: str, rows: list[dict], on_conflict: Literal['do_nothing'] = None,
                    **kwargs) -> int | None:
        ...

    def select_rows(self, table_name: str, schema: str, column_names: list[str] = None,
                    where_clauses: list | tuple | WhereClause = None, order_by: list[OrderByClause] = None,
                    offset: int = None, limit: int = None, **kwargs) -> tuple[list, list[Row]] | tuple[None, None]:
        ...

    def update_rows(self, table_name: str, schema: str, set_clauses: dict[str, Any],
                    where_clauses: list | tuple | WhereClause = None, **kwargs) -> int | None:
        ...

    def delete_rows(self, table_name: str, schema: str, where_clauses: list | tuple | WhereClause = None,
                    **kwargs) -> int | None:
        ...

    # Execute SQL
    def _execute_sql(self, sql: str, return_result: bool = False) -> tuple[list[str], list[Row]] | None:
        ...
