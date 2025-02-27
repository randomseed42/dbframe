import os
from typing import Any, Literal, Type

import pandas as pd
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, DateTime, MetaData, Row, Table, create_engine, delete, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql.sqltypes import TypeEngine
from sqlalchemy.util import FacadeDict

from .base_handler import BaseDFHandler, BaseHandler
from .types import LogLevelType, PyLiteralType
from .utils import (
    NamingValidator, OrderByClause, WhereClause,
    df_to_sql_columns, order_by_parser, where_clauses_parser
)


class SQLiteHandler(BaseHandler):
    def __init__(
            self,
            db_path: str = None,
            log_name: str = 'Logger',
            log_level: LogLevelType = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        super().__init__(log_name=log_name, log_level=log_level, log_console=log_console, log_file=log_file)
        self.db_path = db_path or ':memory:'
        self.url = self._generate_url(self.db_path)
        self.engine = create_engine(
            self.url,
            isolation_level='AUTOCOMMIT',
            connect_args={'check_same_thread': False},
        )
        self._validate_connection()

    def __del__(self):
        self.engine.dispose()

    def _generate_url(self, db_path: str, **kwargs) -> str:
        url_template = 'sqlite:///{}'
        if db_path == ":memory:":
            return url_template.format(':memory:')
        return url_template.format(os.path.abspath(os.path.normpath(db_path)))

    def _validate_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
                self.logger.info('Connection successful')
        except Exception as err:
            self.logger.error(str(err))
            raise ValueError(f'Connection failed: {self.db_path=}')

    # Database CRUD
    def create_database(self, db_path: str, **kwargs) -> str | None:
        url = self._generate_url(db_path=db_path)
        abs_db_path = os.path.abspath(os.path.normpath(db_path))
        if os.path.exists(abs_db_path):
            self.logger.warning(f'The db_path {abs_db_path} already exists')
            return None
        engine = create_engine(url, isolation_level='AUTOCOMMIT')
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
            self.logger.info('Connection successful')
        engine.dispose()
        return abs_db_path

    def get_database(self, db_path: str, **kwargs) -> str | None:
        abs_db_path = os.path.abspath(os.path.normpath(db_path))
        if not os.path.exists(abs_db_path):
            self.logger.warning(f'The db_path {abs_db_path} does not exists')
            return None
        return abs_db_path

    def drop_database(self, db_path: str, **kwargs) -> str | None:
        abs_db_path = os.path.abspath(os.path.normpath(db_path))
        if not os.path.exists(abs_db_path):
            self.logger.warning(f'The db_path {abs_db_path} does not exists')
            return None
        os.remove(abs_db_path)
        return abs_db_path

    # Table CRUD
    def create_table(self, table_name: str, columns: list[Column], **kwargs) -> Table | None:
        table_name = NamingValidator.table(table_name)
        if self.get_table(table_name=table_name) is not None:
            raise ValueError(f'Table {table_name} already exists')
        if len(columns) == 0:
            raise ValueError(f'Columns {columns} cannot be empty')
        metadata = MetaData()
        for column in columns:
            column.name = NamingValidator.column(column.name)
        table = Table(table_name, metadata, *columns, **kwargs)
        table.create(bind=self.engine, checkfirst=True)
        self.logger.info(f'Table {table_name} created')
        return self.get_table(table_name=table_name, **kwargs)

    def get_table(self, table_name: str, **kwargs) -> Table | None:
        table_name = NamingValidator.table(table_name)
        metadata = MetaData()
        try:
            table = Table(table_name, metadata, autoload_with=self.engine, **kwargs)
            return table
        except NoSuchTableError:
            self.logger.warning(f'Table {table_name} not found')
            return None

    def get_tables(self, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        return metadata.tables

    def rename_table(self, old_table_name: str, new_table_name: str, **kwargs) -> str | None:
        old_table_name = NamingValidator.table(old_table_name)
        if self.get_table(table_name=old_table_name, **kwargs) is None:
            raise ValueError(f'Old table {old_table_name} does not exist')
        new_table_name = NamingValidator.table(new_table_name)
        if self.get_table(table_name=new_table_name, **kwargs) is not None:
            raise ValueError(f'New table {new_table_name} already exists')
        with self.engine.connect() as conn:
            stmt = text(f'ALTER TABLE {old_table_name} RENAME TO {new_table_name};')
            conn.execute(stmt)
            self.logger.info(f'Old table {old_table_name} has been renamed to {new_table_name}')
            return new_table_name

    def drop_table(self, table_name: str, **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        table.drop(bind=self.engine, checkfirst=True)
        self.logger.info(f'Table {table.name} dropped')
        return table.name

    def truncate_table(self, table_name: str, restart: bool = True, **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        with self.engine.connect() as conn:
            conn.execute(text(f'DELETE FROM {table.name};'))
            if restart:
                if self.get_table(table_name='sqlite_sequence') is not None:
                    conn.execute(text(f"DELETE FROM sqlite_sequence WHERE name='{table.name}';"))
            self.logger.info(f'Table {table.name} truncated')
            return table.name

    # Column CRUD
    def add_column(self, table_name: str, column: Column, **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        column.name = NamingValidator.column(column.name)
        if self.get_column(table_name=table_name, column_name=column.name, **kwargs) is not None:
            self.logger.warning(f'Column {column.name} already exists in table {table.name}')
            return None
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.add_column(table_name, column)
            self.logger.info(f'Added column {column.name} to table {table_name}')
            return column.name

    def add_columns(self, table_name: str, columns: list[Column], **kwargs) -> list[str] | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        current_columns = self.get_columns(table_name=table_name, **kwargs)
        new_column_names = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for column in columns:
                column.name = NamingValidator.column(column.name)
                if column.name in current_columns or column.name in new_column_names:
                    self.logger.warning(f'Column {column.name} already exists in table {table.name}')
                    continue
                op.add_column(table.name, column)
                new_column_names.append(column.name)
        self.logger.info(f'Added columns {new_column_names} to table {table.name}')
        return new_column_names

    def get_column(self, table_name: str, column_name: str, **kwargs) -> Column | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        column_name = NamingValidator.column(column_name)
        column = table.columns.get(column_name)
        if column is None:
            self.logger.warning(f'Column {column_name} is not found in table {table.name}')
            return None
        return column

    def get_columns(self, table_name: str, **kwargs) -> dict[str, Column] | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        return {k: v for k, v in table.columns.items()}

    def alter_column(
            self,
            table_name: str,
            old_column_name: str,
            new_column_name: str,
            sql_dtype: TypeEngine | Type[TypeEngine] | PyLiteralType = None,
            **kwargs
    ) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        old_column = self.get_column(table_name=table_name, column_name=old_column_name, **kwargs)
        if old_column is None:
            return None
        new_column_name = NamingValidator.column(new_column_name)
        if old_column.name == new_column_name:
            new_column_name = None
        if sql_dtype is not None:
            self.logger.warning('SQLite does not support altering a table column datatype directly. You need to create a new table and copy data from original table')
            sql_dtype = None
        if new_column_name is None and sql_dtype is None:
            self.logger.info(f'Old column {old_column.name=} {old_column.type=} is same with new column in table {table.name}, column remains unchanged')
            return None
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.alter_column(table.name, column_name=old_column.name, new_column_name=new_column_name, **kwargs)
            self.logger.info(f'Altered column {old_column.name} to {new_column_name} in table {table.name}')
            return new_column_name

    def drop_column(self, table_name: str, column_name: str, **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        column_name = NamingValidator.column(column_name)
        if self.get_column(table_name=table_name, column_name=column_name, **kwargs) is None:
            self.logger.warning(f'Column {column_name} is not found in table {table.name}')
            return None
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_column(table_name, column_name, **kwargs)
            self.logger.info(f'Dropped column {column_name} from table {table_name}')
        return column_name

    def drop_columns(self, table_name: str, column_names: list[str], **kwargs) -> list[str] | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        current_columns = self.get_columns(table_name=table_name, **kwargs)
        column_names = map(NamingValidator.column, column_names)
        dropped_column_names = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for column_name in column_names:
                if column_name not in current_columns:
                    self.logger.warning(f'Column {column_name} does not exist in table {table.name}')
                    continue
                if column_name in dropped_column_names:
                    self.logger.warning(f'Column {column_name} is already dropped from table {table.name}')
                    continue
                op.drop_column(table.name, column_name, **kwargs)
                dropped_column_names.append(column_name)
            self.logger.info(f'Dropped columns {dropped_column_names} from table {table.name}')
            return dropped_column_names

    # Index CRUD
    def create_index(self, table_name: str, column_names: list[str], **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        current_columns = self.get_columns(table_name=table_name, **kwargs)
        column_names = map(NamingValidator.column, column_names)
        index_column_names = []
        for column_name in column_names:
            if column_name not in current_columns:
                raise ValueError(f'Column {column_name} is not in {table.name}')
            if column_name in index_column_names:
                raise ValueError(f'Column {column_name} is duplicated')
            index_column_names.append(column_name)
        idx_name = 'ix_{}_{}'.format(table.name, '_'.join(index_column_names))
        if idx_name in self.get_indexes(table_name=table_name, **kwargs):
            raise ValueError(f'Index {idx_name} already exists')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.create_index(idx_name, table_name, index_column_names, **kwargs)
            self.logger.info(f'Create index {idx_name} successful')
            return idx_name

    def get_indexes(self, table_name: str, **kwargs) -> dict[str, list[str]] | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        indexes = {idx.name: list(idx.columns.keys()) for idx in table.indexes}
        return indexes

    def drop_index(self, table_name: str, column_names: list[str], **kwargs) -> str | None:
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            return None
        current_columns = self.get_columns(table_name=table_name, **kwargs)
        column_names = map(NamingValidator.column, column_names)
        index_column_names = []
        for column_name in column_names:
            if column_name not in current_columns:
                raise ValueError(f'Column {column_name} is not in {table.name}')
            if column_name in index_column_names:
                raise ValueError(f'Column {column_name} is duplicated')
            index_column_names.append(column_name)
        idx_name = 'ix_{}_{}'.format(table.name, '_'.join(index_column_names))
        if idx_name not in self.get_indexes(table_name=table_name, **kwargs):
            raise ValueError(f'Index {idx_name} does not exist')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_index(idx_name, table.name, if_exists=True, **kwargs)
            self.logger.info(f'Drop index {idx_name} successful')
            return idx_name

    # Rows CRUD
    def insert_rows(self, table_name: str, rows: list[dict], on_conflict: Literal['do_nothing'] = None,
                    **kwargs) -> int | None:
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')
        if len(rows) == 0:
            self.logger.warning('Rows are empty')
            return 0
        with self.engine.connect() as conn:
            if on_conflict == 'do_nothing':
                cur = conn.execute(insert(table).values(rows).on_conflict_do_nothing())
            else:
                cur = conn.execute(table.insert(), rows)
            rowcount = cur.rowcount if isinstance(cur.rowcount, int) else cur.rowcount()
            self.logger.info(f'Inserted {rowcount} rows to table {table_name}')
            return rowcount

    def select_rows(self, table_name: str, column_names: list[str] = None,
                    where_clauses: list | tuple | WhereClause = None, order_by: list[OrderByClause] = None,
                    offset: int = None, limit: int = None, **kwargs) -> tuple[list, list[Row]] | tuple[None, None]:
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        current_columns = self.get_columns(table_name=table_name, **kwargs)
        if column_names is None:
            selected_columns = current_columns.values()
        else:
            column_names = map(NamingValidator.column, column_names)
            selected_columns = []
            for column_name in column_names:
                if column_name not in current_columns:
                    self.logger.warning(f'Column {column_name} is not in table {table.name}')
                    continue
                selected_columns.append(current_columns.get(column_name))
        if len(selected_columns) == 0:
            return None, None

        where_condition = where_clauses_parser(where_clauses=where_clauses, columns=current_columns)
        orders = order_by_parser(order_by=order_by, columns=current_columns)

        with self.engine.connect() as conn:
            stmt = select(*selected_columns).where(where_condition).order_by(*orders).offset(offset).limit(limit)
            cur = conn.execute(stmt)
            cols = list(cur.keys())
            rows = list(cur.fetchall())
            return cols, rows

    def update_rows(self, table_name: str, set_clauses: dict[str, Any],
                    where_clauses: list | tuple | WhereClause = None, **kwargs) -> int | None:
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        current_columns = self.get_columns(table_name=table_name, **kwargs)
        _set_clauses = {}
        for column_name, set_value in set_clauses.items():
            column_name = NamingValidator.column(column_name)
            if column_name not in current_columns:
                self.logger.warning(f'Column {column_name} is not in {table_name}')
                continue
            if column_name in _set_clauses:
                self.logger.warning(f'Column {column_name} is duplicated in {table_name}')
                continue
            _set_clauses[column_name] = set_value
        if len(_set_clauses) == 0:
            self.logger.warning('Set clauses columns are invalid')
            return None

        where_condition = where_clauses_parser(where_clauses=where_clauses, table=table)

        with self.engine.connect() as conn:
            stmt = table.update().where(where_condition)
            cur = conn.execute(stmt, _set_clauses)
            rowcount = cur.rowcount if isinstance(cur.rowcount, int) else cur.rowcount()
            self.logger.info(f'Updated {rowcount} rows in table {table.name}')
            return rowcount

    def delete_rows(self, table_name: str, where_clauses: list | tuple | WhereClause = None, **kwargs) -> int | None:
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        where_condition = where_clauses_parser(where_clauses=where_clauses, table=table)

        with self.engine.connect() as conn:
            stmt = delete(table).where(where_condition)
            cur = conn.execute(stmt)
            rowcount = cur.rowcount if isinstance(cur.rowcount, int) else cur.rowcount()
            self.logger.info(f'Deleted {rowcount} rows from table {table.name}')
            return rowcount

    # Execute SQL
    def _execute_sql(self, sql: str, return_result: bool = False) -> tuple[list[str], list[Row]] | None:
        with self.engine.connect() as conn:
            stmt = text(sql)
            cur = conn.execute(stmt)
            if not return_result:
                self.logger.info('Execute sql successful')
                return None
            cols = list(cur.keys())
            rows = list(cur.fetchall())
            self.logger.info('Execute sql successful')
            return cols, rows


class SQLiteDFHandler(SQLiteHandler, BaseDFHandler):
    def __init__(
            self,
            db_path: str = None,
            log_name: str = 'Logger',
            log_level: LogLevelType = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        super().__init__(
            db_path=db_path,
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        )

    def df_create_table(
            self,
            df: pd.DataFrame,
            table_name: str,
            primary_column_name: str | Literal['index'] = None,
            primary_sql_column_name: str = None,
            primary_column_autoincrement: bool | Literal['auto'] = 'auto',
            notnull_column_names: list[str] = None,
            index_column_names: list[str | list[str]] = None,
            unique_column_names: list[str | list[str]] = None,
            insert_rows: bool = True,
            convert_df: bool = True,
            **kwargs
    ) -> Table:
        if convert_df:
            df = df.convert_dtypes()
        columns = df_to_sql_columns(
            df=df,
            table_name=table_name,
            primary_column_name=primary_column_name,
            primary_sql_column_name=primary_sql_column_name,
            primary_column_autoincrement=primary_column_autoincrement,
            notnull_column_names=notnull_column_names,
            index_column_names=index_column_names,
            unique_column_names=unique_column_names,
            **kwargs
        )
        table = self.create_table(table_name=table_name, columns=columns, **kwargs)
        if insert_rows:
            self.df_insert_rows(df=df, table_name=table_name, add_columns=False, on_conflict=None, convert_df=False)
        return table

    def df_add_columns(
            self,
            df: pd.DataFrame,
            table_name: str,
            convert_df: bool = True,
            **kwargs
    ) -> list[str]:
        if convert_df:
            df = df.convert_dtypes()
        df = df.rename(columns=NamingValidator.column)
        current_columns = self.get_columns(table_name=table_name)
        new_column_names = []
        for column_name in df.columns:
            if column_name in current_columns:
                continue
            new_column_names.append(column_name)
        new_columns = df_to_sql_columns(df=df[new_column_names], table_name=table_name, **kwargs)
        added_column_names = self.add_columns(table_name=table_name, columns=new_columns)
        return added_column_names

    def df_alter_columns_type(
            self,
            df: pd.DataFrame,
            table_name: str,
            convert_df: bool = True,
            **kwargs
    ):
        raise ValueError('SQLite does not support alter a table column datatype directly. You need to create a new table and copy data from original table')

    def df_insert_rows(
            self,
            df: pd.DataFrame,
            table_name: str,
            add_columns: bool = True,
            on_conflict: Literal['do_nothing'] = None,
            convert_df: bool = True,
            **kwargs
    ) -> int | None:
        if convert_df:
            df = df.convert_dtypes()
        df = df.rename(columns=NamingValidator.column)
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')
        if add_columns:
            self.df_add_columns(df=df, table_name=table_name, convert_df=False)
        current_columns = self.get_columns(table_name=table_name)
        subset_columns = [col for col in df.columns if col in current_columns]
        for col in subset_columns:
            if isinstance(current_columns[col].type, DateTime):
                df[col] = pd.to_datetime(df[col])
        rows = df[subset_columns].where(pd.notnull(df[subset_columns]), None).to_dict(orient='records')
        rowcount = self.insert_rows(table_name=table.name, rows=rows, on_conflict=on_conflict, **kwargs)
        return rowcount

    def df_select_rows(self, table_name: str, column_names: list[str] = None, where_clauses: list | tuple | WhereClause = None, order_by: list[OrderByClause] = None, offset: int = None, limit: int = None, **kwargs) -> pd.DataFrame | None:
        cols, rows = self.select_rows(table_name=table_name, column_names=column_names, where_clauses=where_clauses, order_by=order_by, offset=offset, limit=limit, **kwargs)
        if cols is None or rows is None:
            return None
        df = pd.DataFrame(data=rows, columns=cols)
        return df


