import os
from typing import Literal, Any

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text, MetaData, Table, Column, select, Row, delete
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError

from .logger import Logger
from .utils import WhereClause, OrderByClause, where_clauses_parser, order_by_parser


class SQLiteHandler:
    def __init__(
            self,
            db_path: str = None,
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        self.db_path = db_path
        self.logger = Logger(
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        ).get_logger()
        self.logger.debug(f'logger initialized {self.logger.name=} {id(self.logger)=}')

        self.url = self._generate_url()
        self.engine = create_engine(
            self.url,
            isolation_level='AUTOCOMMIT',
        )
        self._validate_connection()

    def _generate_url(self) -> str:
        url_template = 'sqlite:///{}'
        if self.db_path is None or self.db_path == ":memory:":
            return url_template.format(':memory:')
        return url_template.format(os.path.abspath(self.db_path))

    def _validate_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
                self.logger.info('Connection successful')
        except Exception as err:
            self.logger.error(str(err))
            raise ValueError(f'Connection failed: {self.db_path=}')

    def get_tables(self, **kwargs) -> dict[str, Table]:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, **kwargs)
        tables = metadata.tables
        return tables

    def get_table(self, table_name: str, **kwargs) -> Table | None:
        metadata = MetaData()
        try:
            table = Table(table_name, metadata, autoload_with=self.engine, **kwargs, )
            return table
        except NoSuchTableError:
            self.logger.info(f'Table {table_name} not found')
            return None

    def create_table(self, table_name: str, columns: list[Column], **kwargs) -> Table | None:
        if self.get_table(table_name) is not None:
            self.logger.warning(f'Table {table_name} already exists')
            return self.get_table(table_name, **kwargs)
        metadata = MetaData()
        table = Table(table_name, metadata, *columns, **kwargs)
        table.create(bind=self.engine, checkfirst=True)
        self.logger.info(f'Table {table_name} created')
        return self.get_table(table_name, **kwargs)

    def drop_table(self, table_name: str, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is not None:
            table.drop(bind=self.engine, checkfirst=True)
            self.logger.info(f'Table {table_name} dropped')

    def get_column(self, table_name: str, column: str, **kwargs) -> Column | None:
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return None
        return table.columns.get(column)

    def get_columns(self, table_name: str, **kwargs) -> dict[str, Column] | None:
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return None
        return {k: v for k, v in table.columns.items()}

    def add_column(self, table_name: str, column: Column, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        current_columns = self.get_columns(table_name, **kwargs)
        if column.name in current_columns:
            return
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.add_column(table_name, column)
            self.logger.info(f'Added column {column.name} to table {table_name}')

    def add_columns(self, table_name: str, columns: list[Column], **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        current_columns = self.get_columns(table_name, **kwargs)
        new_columns = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for column in columns:
                if column.name in current_columns:
                    self.logger.warning(f'Column {column.name} already exists in table {table_name}')
                    continue
                op.add_column(table_name, column, **kwargs)
                new_columns.append(column.name)
            self.logger.info(f'Added columns {new_columns} to table {table_name}')

    def drop_column(self, table_name: str, column: str, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        current_columns = self.get_columns(table_name, **kwargs)
        if column not in current_columns:
            return
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_column(table_name, column, **kwargs)
            self.logger.info(f'Dropped column {column} from table {table_name}')

    def drop_columns(self, table_name: str, columns: list[str], **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        current_columns = self.get_columns(table_name, **kwargs)
        dropped_columns = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for column in columns:
                if column not in current_columns:
                    self.logger.warning(f'Column {column} is not in table {table_name}')
                    continue
                op.drop_column(table_name, column, **kwargs)
                dropped_columns.append(column)
            self.logger.info(f'Dropped columns {dropped_columns} from table {table_name}')

    def alter_column(self, table_name: str, column: str, new_column_name: str, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        old_column = self.get_column(table_name, column=column, **kwargs)
        if old_column is None:
            return
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.alter_column(table_name, column_name=column, new_column_name=new_column_name, **kwargs)
            self.logger.info(f'Altered column {column} to {new_column_name=} in table {table_name}')

    def get_indexes(self, table_name: str, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        indexes = {idx.name: list(idx.columns.keys()) for idx in table.indexes}
        return indexes

    def create_index(self, table_name: str, columns: list[str], **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        current_columns = self.get_columns(table_name, **kwargs)
        for column in columns:
            if column not in current_columns:
                raise ValueError(f'Column {column} is not in {table_name}')
        idx_name = 'ix_{}_{}'.format(table_name, '_'.join(columns))
        if idx_name in self.get_indexes(table_name=table_name, **kwargs):
            raise ValueError(f'Index {idx_name} already exists')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.create_index(idx_name, table_name, columns, **kwargs)
            self.logger.info(f'Create index {idx_name} successful')

    def drop_index(self, table_name: str, columns: list[str], **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            return
        idx_name = 'ix_{}_{}'.format(table_name, '_'.join(columns))
        if idx_name not in self.get_indexes(table_name=table_name, **kwargs):
            raise ValueError(f'Index {idx_name} does not exist')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_index(idx_name, table_name, if_exists=True, **kwargs)
            self.logger.info(f'Drop index {idx_name} successful')

    def select_rows(
            self,
            table_name: str,
            columns: list[str] = None,
            where_clauses: list | tuple | WhereClause = None,
            order_by: list[OrderByClause] = None,
            offset: int = None,
            limit: int = None,
            **kwargs
    ) -> tuple[list, list[Row]] | tuple[None, None]:
        table = self.get_table(table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        current_columns = self.get_columns(table_name, **kwargs)
        if columns is None:
            selected_columns = current_columns.values()
        else:
            selected_columns = []
            for col in columns:
                if col not in current_columns:
                    self.logger.warning(f'Column {col} is not in table {table_name}')
                    continue
                selected_columns.append(current_columns.get(col))
        if len(selected_columns) == 0:
            return None, None

        where_condition = where_clauses_parser(where_clauses=where_clauses, columns=current_columns)
        orders = order_by_parser(order_by=order_by, columns=current_columns)

        stmt = select(*selected_columns).where(where_condition).order_by(*orders).offset(offset).limit(limit)

        with self.engine.connect() as conn:
            cur = conn.execute(stmt)
            cols = list(cur.keys())
            rows = list(cur.fetchall())
            return cols, rows

    def insert_rows(self, table_name: str, rows: list[dict], on_conflict: Literal['do_nothing'] = None, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')
        if len(rows) == 0:
            self.logger.warning(f'Rows are empty')
            return
        with self.engine.connect() as conn:
            if on_conflict == 'do_nothing':
                conn.execute(insert(table).values(rows).on_conflict_do_nothing())
            else:
                conn.execute(table.insert(), rows)
            self.logger.info(f'Inserted {len(rows)} rows')

    def delete_rows(self, table_name: str, where_clauses: list | tuple | WhereClause = None, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        where_condition = where_clauses_parser(where_clauses=where_clauses, table=table)

        stmt = delete(table).where(where_condition)

        with self.engine.connect() as conn:
            cur = conn.execute(stmt)
            self.logger.info(f'Deleted {cur.rowcount} rows')

    def update_rows(self, table_name: str, set_clauses: dict[str, Any],
                    where_clauses: list | tuple | WhereClause = None, **kwargs):
        table = self.get_table(table_name, **kwargs)
        if table is None:
            raise ValueError(f'Table {table_name} not found')

        where_condition = where_clauses_parser(where_clauses=where_clauses, table=table)
        stmt = table.update().where(where_condition)
        with self.engine.connect() as conn:
            cur = conn.execute(stmt, set_clauses)
            self.logger.info(f'Updated {cur.rowcount} rows')

    def _execute_sql(self, sql: str, return_result: bool = False) -> tuple[list[str], list[Row]] | None:
        stmt = text(sql)
        with self.engine.connect() as conn:
            cur = conn.execute(stmt)
            if not return_result:
                self.logger.info('Execute sql successful')
                return None
            cols = list(cur.keys())
            rows = list(cur.fetchall())
            self.logger.info('Execute sql successful')
            return cols, rows


class SQLiteDFHandler(SQLiteHandler):
    def __init__(
            self,
            db_path: str = None,
            log_name: str = 'Logger',
            log_level: str = None,
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
