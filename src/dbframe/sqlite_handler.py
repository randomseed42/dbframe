import os
from typing import Literal

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text, MetaData, Table, Column
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError

from .logger import Logger


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
        self.logger.critical(f'logger initialized {self.logger.name=} {id(self.logger)=}')

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
