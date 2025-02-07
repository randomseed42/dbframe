import os
from typing import Literal, Any
from urllib.parse import quote_plus

import psycopg2
from alembic.migration import MigrationContext
from alembic.operations import Operations
from psycopg2.errors import BadCopyFileFormat
from sqlalchemy import create_engine, text, MetaData, Table, Column, select, Row, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DataError, NoSuchTableError
from sqlalchemy.util import FacadeDict

from .logger import Logger
from .utils import NamingValidator


class PGHandler:
    def __init__(
            self,
            host: str = None,
            port: int | str = None,
            user: str = None,
            password: str = None,
            dbname: str = None,
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        self.host = host or os.getenv('PG_HOST', '127.0.0.1')
        self.port = port or os.getenv('PG_PORT', 5432)
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = password or os.getenv('PG_PASS', 'postgres')
        self.dbname = dbname or os.getenv('PG_DBNAME', 'postgres')
        self.dbname = NamingValidator.dbname(self.dbname)
        self.logger = Logger(
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        ).get_logger()
        self.logger.debug(f'logger initialized {self.logger.name=} {id(self.logger)=}')

        self.url = self._generate_url(dbname=self.dbname)
        self.url_default = self._generate_url(dbname='postgres')
        self.engine = create_engine(
            self.url,
            executemany_mode='values_plus_batch',
            isolation_level='AUTOCOMMIT',
        )
        self._validate_connection()

    def __del__(self):
        self.engine.dispose()

    def _generate_url(self, dbname: str):
        url_template = 'postgresql+psycopg2://{}:{}@{}:{}/{}'
        return url_template.format(self.user, quote_plus(self.password), self.host, self.port, dbname)

    def _validate_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
                self.logger.info('Connection successful')
        except Exception as err:
            self.logger.error(str(err))
            raise ValueError(f'Connection failed: {self.host=} {self.port=} {self.user=} {self.dbname=}')

    def get_databases(self):
        stmt = text('SELECT datname FROM pg_catalog.pg_database;')
        engine = create_engine(self.url_default)
        with engine.connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchall()
            databases = [row[0] for row in rows]
            return databases

    def create_database(self, dbname: str) -> str | None:
        dbname = NamingValidator.dbname(dbname)
        current_databases = self.get_databases()
        if dbname in current_databases:
            self.logger.warning(f'The dbname {dbname} already exist')
            return None
        with self.engine.connect() as conn:
            stmt = text(f"CREATE DATABASE {dbname} LOCALE 'en_US.utf8' ENCODING UTF8 TEMPLATE template0;")
            conn.execute(stmt)
            self.logger.info(f'Database {dbname} created')
            return dbname

    def drop_database(self, dbname: str) -> str | None:
        dbname = NamingValidator.dbname(dbname)
        current_databases = self.get_databases()
        if dbname not in current_databases:
            self.logger.warning(f'The dbname {dbname} does not exist')
            return None
        with self.engine.connect() as conn:
            stmt = text(f'DROP DATABASE IF EXISTS {dbname};')
            conn.execute(stmt)
            self.logger.info(f'Database {dbname} dropped')
            return dbname

    def get_schemas(self) -> list[str]:
        stmt = text('SELECT nspname FROM pg_catalog.pg_namespace;')
        with self.engine.connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchall()
            schemas = [row[0] for row in rows]
            return schemas

    def create_schema(self, schema: str) -> str | None:
        schema = NamingValidator.schema(schema)
        current_schemas = self.get_schemas()
        if schema in current_schemas:
            self.logger.warning(f'The schema {schema} already exist')
            return None
        with self.engine.connect() as conn:
            stmt = text(f'CREATE SCHEMA IF NOT EXISTS {schema};')
            conn.execute(stmt)
            self.logger.info(f'Schema {schema} created')
            return schema

    def drop_schema(self, schema: str) -> str | None:
        schema = NamingValidator.schema(schema)
        current_schemas = self.get_schemas()
        if schema not in current_schemas:
            self.logger.warning(f'The schema {schema} does not exist')
            return None
        with self.engine.connect() as conn:
            stmt = text(f'DROP SCHEMA IF EXISTS {schema};')
            conn.execute(stmt)
            self.logger.info(f'Schema {schema} dropped')
            return schema

    def get_tables(self, schema: str = 'public', views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict:
        schema = NamingValidator.schema(schema)
        schemas = self.get_schemas()
        if schema not in schemas:
            raise ValueError(f'Schema {schema} does not exist in database {self.dbname}')
        metadata = MetaData(schema=schema)
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        tables = metadata.tables
        return tables

    def get_table(self, table_name: str, schema: str = 'public', **kwargs) -> Table | None:
        table_name = NamingValidator.table(table_name)
        schema = NamingValidator.schema(schema)
        schemas = self.get_schemas()
        if schema not in schemas:
            raise ValueError(f'Schema {schema} does not exist in database {self.dbname}')
        metadata = MetaData(schema=schema)
        try:
            table = Table(table_name, metadata, autoload_with=self.engine, **kwargs)
            return table
        except NoSuchTableError:
            self.logger.info(f'Table {table_name} not found in schema {schema}')
            return None

    def create_table(self, table_name: str, columns: list[Column], schema: str, **kwargs):
        schema = NamingValidator.schema(schema)
        current_schemas = self.get_schemas()
        if schema not in current_schemas:
            self.logger.warning(f'The schema {schema} does not exist')
            self.create_schema(schema=schema)
        table_name = NamingValidator.table(table_name)
        if self.get_table(table_name=table_name, schema=schema) is not None:
            raise ValueError(f'Table {table_name} already exists in schema {schema}')
        metadata = MetaData(schema=schema)
        table = Table(table_name, metadata, *columns, **kwargs)
        table.create(bind=self.engine, checkfirst=True)
        self.logger.info(f'Table {table_name} created in schema {schema}')
        return self.get_table(table_name, schema, **kwargs)

    def drop_table(self, table_name: str, schema: str, **kwargs):
        schema = NamingValidator.schema(schema)
        current_schemas = self.get_schemas()
        if schema not in current_schemas:
            raise ValueError(f'The schema {schema} does not exist')
        table_name = NamingValidator.table(table_name)
        table = self.get_table(table_name=table_name, schema=schema, **kwargs)
        if table is None:
            self.logger.warning(f'Table {table_name} does not exists in schema {schema}')
            return
        table.drop(bind=self.engine, checkfirst=True)
        self.logger.info(f'Table {schema}.{table_name} dropped')

    def rename_table(self, old_table_name: str, new_table_name: str, schema: str, **kwargs):
        schema = NamingValidator.schema(schema)
        current_schemas = self.get_schemas()
        if schema not in current_schemas:
            raise ValueError(f'The schema {schema} does not exist')
        old_table_name = NamingValidator.table(old_table_name)
        if self.get_table(table_name=old_table_name, schema=schema, **kwargs) is None:
            raise ValueError(f'Old table {old_table_name} does not exist in schema {schema}')
        new_table_name = NamingValidator.table(new_table_name)
        if self.get_table(table_name=new_table_name, schema=schema, **kwargs) is not None:
            raise ValueError(f'New table {new_table_name} already exists in schema {schema}')
        with self.engine.connect() as conn:
            stmt = text(f'ALTER TABLE {schema}.{old_table_name} RENAME TO {new_table_name};')
            conn.execute(stmt)
            self.logger.info(f'Old table {old_table_name} has been renamed to {new_table_name} in schema {schema}')


class PGDFHandler(PGHandler):
    def __init__(
            self,
            host: str = None,
            port: int | str = None,
            user: str = None,
            password: str = None,
            dbname: str = None,
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        super().__init__(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        )
