import os
from io import BytesIO
from typing import Any, Literal, Sequence
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    JSON,
    Column,
    Constraint,
    CursorResult,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    Row,
    Table,
    TextClause,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import ProgrammingError, ResourceClosedError
from sqlalchemy.sql.sqltypes import TypeEngine
from sqlalchemy.util import FacadeDict

from .clause import Order, Where, order_parser, where_parser
from .dtype import df_to_rows, df_to_schema_items, object_to_sql_dtype
from .validator import NameValidator


class Pgsql:
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
            execution_options={
                'executemany_mode': 'values',
                'executemany_values_page_size': 1000,
            },
            **kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __del__(self) -> None:
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

    def dispose(self):
        self.engine.dispose()

    # Database CRUD
    def create_database(self, dbname: str) -> str:
        dbname = NameValidator.dbname(dbname)
        pg = Pgsql(
            host=self.host, port=self.port, user=self.user, password=self.password, dbname='postgres', verbose=self.verbose
        )
        if pg.get_database(dbname=dbname) is not None:
            raise ValueError(f'Database {dbname} already exists.')
        stmt = text(f"CREATE DATABASE {dbname} LOCALE 'en_US.utf8' ENCODING UTF8 TEMPLATE template0;")
        pg._execute_sql(stmt)
        self._verbose_print(f'Created database {dbname}.')
        return dbname

    def get_database(self, dbname: str) -> str | None:
        dbname = NameValidator.dbname(dbname)
        pg = Pgsql(
            host=self.host, port=self.port, user=self.user, password=self.password, dbname='postgres', verbose=self.verbose
        )
        # _, rows = pg.select_rows(schema_nm='pg_catalog', tb_nm='pg_database', col_nms=['datname'], where=Where('datname', '==', dbname))
        # if len(rows) == 0:
        #     return
        # return rows[0][0]
        stmt = text('SELECT datname FROM pg_catalog.pg_database WHERE datname = :dbname;').bindparams(dbname=dbname)
        cur = pg._execute_sql(stmt)
        row = cur.fetchone()
        if row is None:
            return
        return row[0]

    def get_databases(self) -> Sequence[str] | None:
        pg = Pgsql(
            host=self.host, port=self.port, user=self.user, password=self.password, dbname='postgres', verbose=self.verbose
        )
        stmt = text('SELECT datname FROM pg_catalog.pg_database;')
        cur = pg._execute_sql(stmt)
        rows = cur.fetchall()
        return [r[0] for r in rows]

    def drop_database(self, dbname: str) -> str:
        dbname = self.get_database(dbname)
        if dbname is None:
            raise ValueError(f'Database {dbname} does not exist.')
        if dbname == self.dbname:
            raise ValueError(f'Cannot drop the current database {dbname}.')
        pg = Pgsql(
            host=self.host, port=self.port, user=self.user, password=self.password, dbname='postgres', verbose=self.verbose
        )
        stmt = text(f'DROP DATABASE {dbname};')
        pg._execute_sql(stmt)
        self._verbose_print(f'Dropped database {dbname}.')
        return dbname

    # Schema CRUD
    def create_schema(self, schema_nm: str) -> str | None:
        schema_nm = NameValidator.schema(schema_nm)
        if inspect(self.engine).has_schema(schema_name=schema_nm):
            raise ValueError(f'Schema {schema_nm} already exists.')
        stmt = text(f'CREATE SCHEMA IF NOT EXISTS {schema_nm};')
        self._execute_sql(stmt)
        self._verbose_print(f'Created schema {schema_nm}.')
        return schema_nm

    def get_schema(self, schema_nm: str) -> str | None:
        schema_nm = NameValidator.schema(schema_nm)
        stmt = text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_nm}';")
        cur = self._execute_sql(stmt)
        row = cur.fetchone()
        if row is None:
            return
        return row[0]

    def get_schemas(self) -> Sequence[str] | None:
        stmt = text('SELECT schema_name FROM information_schema.schemata;')
        cur = self._execute_sql(stmt)
        rows = cur.fetchall()
        return [r[0] for r in rows]

    def drop_schema(self, schema_nm: str, cascade: bool = False) -> str | None:
        schema_nm = NameValidator.schema(schema_nm)
        if self.get_schema(schema_nm=schema_nm) is None:
            raise ValueError(f'Schema {schema_nm} does not exist.')
        stmt = text(f'DROP SCHEMA {schema_nm} {"CASCADE" if cascade else ""};')
        self._execute_sql(stmt)
        self._verbose_print(f'Dropped schema {schema_nm}.')
        return schema_nm

    # Table CRUD
    def create_table(
        self,
        schema_nm: str,
        tb_nm: str,
        cols: Sequence[Column | Constraint | Index | UniqueConstraint],
        **kwargs,
    ) -> Table:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        if self.table_exists(schema_nm=schema_nm, tb_nm=tb_nm):
            raise ValueError(f'Table {schema_nm}.{tb_nm} already exists.')
        if len(cols) == 0:
            raise ValueError(f'Columns {cols} cannot be empty.')
        for col in cols:
            col.name = NameValidator.column(col.name)
        metadata = MetaData(schema=schema_nm)
        for col in cols:
            col.name = NameValidator.column(col.name)
        tb = Table(tb_nm, metadata, *cols)
        tb.create(bind=self.engine, checkfirst=False)
        self._verbose_print(f'Table {schema_nm}.{tb_nm} created.')
        return self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)

    def table_exists(self, schema_nm: str, tb_nm: str) -> bool:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        return inspect(self.engine).has_table(table_name=tb_nm, schema=schema_nm)

    def get_table(self, schema_nm: str, tb_nm: str, **kwargs) -> Table:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        if not self.table_exists(schema_nm=schema_nm, tb_nm=tb_nm):
            raise ValueError(f'Table {schema_nm}.{tb_nm} does not exist.')
        metadata = MetaData(schema=schema_nm)
        tb = Table(tb_nm, metadata, autoload_with=self.engine)
        return tb

    def get_tables(self, schema_nm: str, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        schema_nm = NameValidator.schema(schema_nm)
        metadata = MetaData(schema=schema_nm)
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        return metadata.tables

    def rename_table(self, schema_nm: str, tb_nm: str, new_tb_nm: str) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        new_tb_nm = NameValidator.table(new_tb_nm)
        stmt = text(f'ALTER TABLE {schema_nm}.{tb_nm} RENAME TO {new_tb_nm};')
        self._execute_sql(stmt)
        self._verbose_print(f'Table {schema_nm}.{tb_nm} renamed to {new_tb_nm}.')
        return new_tb_nm

    def drop_table(self, schema_nm: str, tb_nm: str, **kwargs) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        tb.drop(bind=self.engine, checkfirst=False)
        self._verbose_print(f'Table {schema_nm}.{tb_nm} dropped.')
        return tb_nm

    def truncate_table(self, schema_nm: str, tb_nm: str, restart: bool = True) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        stmt = text(f'TRUNCATE TABLE {schema_nm}.{tb_nm} {"RESTART IDENTITY" if restart else ""};')
        self._execute_sql(stmt)
        self._verbose_print(f'Table {schema_nm}.{tb_nm} truncated.')
        return tb_nm

    # Column CRUD
    def add_column(self, schema_nm: str, tb_nm: str, col: Column, **kwargs) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col.name)
        col.name = col_nm
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.add_column(tb_nm, col, schema=schema_nm)
            self._verbose_print(f'Column {col_nm} added to table {schema_nm}.{tb_nm}.')
            return col_nm

    def add_columns(self, schema_nm: str, tb_nm: str, cols: Sequence[Column], **kwargs) -> Sequence[str]:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        new_col_nms = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for col in cols:
                col_nm = NameValidator.column(col.name)
                col.name = col_nm
                if col_nm in current_cols or col_nm in new_col_nms:
                    continue
                if col.primary_key:
                    self._verbose_print(f'Column {col_nm} is primary key and cannot be added in table {schema_nm}.{tb_nm}.')
                    continue
                op.add_column(tb_nm, col, schema=schema_nm, **kwargs)
                new_col_nms.append(col_nm)
            if new_col_nms:
                self._verbose_print(f'Columns {new_col_nms} added to table {schema_nm}.{tb_nm}.')
            return new_col_nms

    def get_column(self, schema_nm: str, tb_nm: str, col_nm: str, **kwargs) -> Column:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        col = tb.columns.get(col_nm)
        if col is None:
            raise ValueError(f'Column {col_nm} does not exist in table {schema_nm}.{tb_nm}.')
        return col

    def get_columns(self, schema_nm: str, tb_nm: str, **kwargs) -> dict[str, Column]:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        return {col_nm: col for col_nm, col in tb.columns.items()}

    def rename_column(self, schema_nm: str, tb_nm: str, col_nm: str, new_col_nm: str, **kwargs) -> Column:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        new_col_nm = NameValidator.column(new_col_nm)
        if new_col_nm == col_nm:
            raise ValueError(f'New column name {new_col_nm} is same with old column name {col_nm}.')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.alter_column(tb_nm, col_nm, new_column_name=new_col_nm, schema=schema_nm, **kwargs)
            self._verbose_print(f'Column {col_nm} renamed to {new_col_nm} in table {schema_nm}.{tb_nm}.')
        return self.get_column(schema_nm=schema_nm, tb_nm=tb_nm, col_nm=new_col_nm, **kwargs)

    def alter_column(
        self, schema_nm: str, tb_nm: str, col_nm: str, sql_dtype: str | TypeEngine | type, new_col_nm: str = None, **kwargs
    ) -> Column:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        if new_col_nm is not None:
            new_col_nm = NameValidator.column(new_col_nm)
        if not isinstance(sql_dtype, TypeEngine):
            sql_dtype = object_to_sql_dtype(sql_dtype)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            try:
                if isinstance(sql_dtype, JSON):
                    op.alter_column(
                        tb_nm,
                        col_nm,
                        new_column_name=new_col_nm,
                        type_=sql_dtype,
                        schema=schema_nm,
                        postgresql_using=f'to_jsonb({col_nm}::text)::json',
                        **kwargs,
                    )
                else:
                    op.alter_column(tb_nm, col_nm, new_column_name=new_col_nm, type_=sql_dtype, schema=schema_nm, **kwargs)
                self._verbose_print(
                    f'Column {col_nm} altered to {new_col_nm if new_col_nm else col_nm} type {sql_dtype} in table {schema_nm}.{tb_nm}.'
                )
            except ProgrammingError:
                self._verbose_print(
                    f'Column {col_nm} cannot be altered to {new_col_nm if new_col_nm else col_nm} type {sql_dtype} in table {schema_nm}.{tb_nm}.'
                )
        return self.get_column(schema_nm=schema_nm, tb_nm=tb_nm, col_nm=new_col_nm or col_nm, **kwargs)

    def drop_column(self, schema_nm: str, tb_nm: str, col_nm: str, **kwargs) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_column(tb_nm, col_nm, schema=schema_nm, **kwargs)
            self._verbose_print(f'Column {col_nm} dropped from table {schema_nm}.{tb_nm}.')
            return col_nm

    def drop_columns(self, schema_nm: str, tb_nm: str, col_nms: Sequence[str], **kwargs) -> Sequence[str]:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        dropped_col_nms = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    self._verbose_print(f'Column {col_nm} does not exist in table {schema_nm}.{tb_nm}.')
                    continue
                op.drop_column(tb_nm, col_nm, schema=schema_nm, **kwargs)
                dropped_col_nms.append(col_nm)
            self._verbose_print(f'Columns {dropped_col_nms} dropped from table {schema_nm}.{tb_nm}.')
            return dropped_col_nms

    # Index CRUD
    def create_index(
        self, schema_nm: str, tb_nm: str, col_nms: Sequence[str], idx_nm: str = None, unique: bool = False, **kwargs
    ) -> Index:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
        current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        index_col_nms = []
        for col_nm in col_nms:
            if col_nm not in current_cols:
                raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
            if col_nm in index_col_nms:
                raise ValueError(f'Column {col_nm} is duplicated in {col_nms}.')
            index_col_nms.append(col_nm)
        idx_nm = NameValidator.index(idx_nm or f'idx_{tb_nm}_{"_".join(index_col_nms)}')
        if idx_nm in self.get_indexes(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs):
            raise ValueError(f'Index {idx_nm} already exists.')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.create_index(idx_nm, tb_nm, col_nms, unique=unique, schema=schema_nm, **kwargs)
            self._verbose_print(f'Index {idx_nm} created on table {schema_nm}.{tb_nm} with columns {col_nms}.')
        return self.get_index(schema_nm=schema_nm, tb_nm=tb_nm, idx_nm=idx_nm, **kwargs)

    def get_index(self, schema_nm: str, tb_nm: str, col_nms: Sequence[str] = None, idx_nm: str = None, **kwargs) -> Index:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        if idx_nm is None and col_nms is None:
            raise ValueError('Either idx_nm or col_nms must be provided.')
        indexes = self.get_indexes(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if idx_nm is not None:
            if idx_nm in indexes:
                return indexes.get(idx_nm)
            raise ValueError(f'Index {idx_nm} does not exist in table {schema_nm}.{tb_nm}.')
        if col_nms is not None:
            col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
            current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
            for col_nm in col_nms:
                if col_nm not in current_cols:
                    raise ValueError(f'Column {col_nm} does not exist in table {schema_nm}.{tb_nm}.')
            for idx_nm, idx in indexes.items():
                if set(idx.columns.keys()) == set(col_nms):
                    return idx
            raise ValueError(f'Index with columns {col_nms} does not exist in table {schema_nm}.{tb_nm}.')

    def get_indexes(self, schema_nm: str, tb_nm: str, **kwargs) -> dict[str, Index]:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        return {idx.name: idx for idx in tb.indexes}

    def drop_index(self, schema_nm: str, tb_nm: str, col_nms: Sequence[str] = None, idx_nm: str = None, **kwargs) -> str:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        idx = self.get_index(schema_nm=schema_nm, tb_nm=tb_nm, col_nms=col_nms, idx_nm=idx_nm, **kwargs)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_index(idx.name, tb_nm, schema=schema_nm, if_exists=False, **kwargs)
            self._verbose_print(f'Index {idx.name} dropped from table {schema_nm}.{tb_nm}.')
            return idx.name

    # Row CRUD
    def insert_row(
        self,
        schema_nm: str,
        tb_nm: str,
        row: dict[str, Any],
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if row_key_validate:
            row = {NameValidator.column(key): val for key, val in row.items()}
        with self.engine.connect() as conn:
            if on_conflict in ('update', 'replace', 'upsert'):
                stmt = insert(tb).values(**row).on_conflict_do_update(index_elements=tb.primary_key.columns, set_=row)
                with conn.begin():
                    cur = conn.execute(stmt)
                self._verbose_print(f'Inserted or replaced row into table {schema_nm}.{tb_nm}.')
            elif on_conflict in ('do_nothing', 'ignore', 'skip'):
                stmt = insert(tb).values(**row).on_conflict_do_nothing()
                with conn.begin():
                    cur = conn.execute(stmt)
                self._verbose_print(f'Inserted or ignored row into table {schema_nm}.{tb_nm}.')
            elif on_conflict is None:
                stmt = insert(tb).values(**row)
                with conn.begin():
                    cur = conn.execute(stmt)
                self._verbose_print(f'Inserted row into table {schema_nm}.{tb_nm}.')
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}.')
            return cur.rowcount

    def insert_rows(
        self,
        schema_nm: str,
        tb_nm: str,
        rows: Sequence[dict[str, Any]],
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if row_key_validate:
            rows = [{NameValidator.column(key): val for key, val in row.items()} for row in rows]
        with self.engine.connect() as conn:
            if on_conflict in ('update', 'replace', 'upsert'):
                stmt = insert(tb)
                uq_constraints_col_nms = []
                pk_constraints_col_nms = []
                if len(tb.constraints) > 0:
                    for constraint in tb.constraints:
                        if isinstance(constraint, UniqueConstraint):
                            uq_constraints_col_nms.extend(constraint.columns.keys())
                        elif isinstance(constraint, PrimaryKeyConstraint):
                            pk_constraints_col_nms.extend(constraint.columns.keys())
                constraints_col_nms = uq_constraints_col_nms or pk_constraints_col_nms
                if len(constraints_col_nms) > 0:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=constraints_col_nms,
                        set_={
                            col_nm: getattr(stmt.excluded, col_nm)
                            for col_nm in tb.columns.keys()
                            if col_nm not in constraints_col_nms and col_nm not in tb.primary_key.columns.keys()
                        },
                    )
                with conn.begin():
                    cur = conn.execute(stmt, rows)
                self._verbose_print(f'Inserted or replaced rows into table {schema_nm}.{tb_nm}.')
            elif on_conflict in ('do_nothing', 'ignore', 'skip'):
                stmt = insert(tb).on_conflict_do_nothing()
                with conn.begin():
                    cur = conn.execute(stmt, rows)
                self._verbose_print(f'Inserted or ignored rows into table {schema_nm}.{tb_nm}.')
            elif on_conflict is None:
                stmt = insert(tb)
                with conn.begin():
                    cur = conn.execute(stmt, rows)
                self._verbose_print(f'Inserted rows into table {schema_nm}.{tb_nm}.')
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}.')
            return cur.rowcount

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
    ) -> tuple[list[str], list[Row]] | None:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        # self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if col_nms is None:
            selected_cols = list(current_cols.values())
        else:
            selected_cols = []
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    raise ValueError(f'Column {col_nm} does not exist in table {schema_nm}.{tb_nm}.')
                selected_cols.append(current_cols[col_nm])
        if len(selected_cols) == 0:
            raise ValueError('No columns selected.')

        where_cond = where_parser(where, cols=current_cols, tb=None)
        order_cond = order_parser(order, cols=current_cols, tb=None)
        with self.engine.connect() as conn:
            stmt = select(*selected_cols).where(where_cond).order_by(*order_cond).offset(offset).limit(limit)
            cur = conn.execute(stmt)
            cols = list(cur.keys())
            rows = cur.fetchall()
            self._verbose_print(f'Selected {len(rows)} rows from table {schema_nm}.{tb_nm}.')
            return cols, rows

    def update_rows(
        self, schema_nm: str, tb_nm: str, set_values: dict[str, Any], where: Where | Sequence = None, **kwargs
    ) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        _set_values = {}
        for col_nm, val in set_values.items():
            col_nm = NameValidator.column(col_nm)
            if col_nm not in tb.columns:
                raise ValueError(f'Column {col_nm} does not exist in table {schema_nm}.{tb_nm}.')
            _set_values[col_nm] = val
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.update().where(where_cond)
            with conn.begin():
                cur = conn.execute(stmt, _set_values)
            self._verbose_print(f'Updated {cur.rowcount} rows in table {schema_nm}.{tb_nm}.')
            return cur.rowcount

    def delete_rows(self, schema_nm: str, tb_nm: str, where: Where | Sequence = None, **kwargs) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.delete().where(where_cond)
            with conn.begin():
                cur = conn.execute(stmt)
            self._verbose_print(f'Deleted {cur.rowcount} rows from table {schema_nm}.{tb_nm}.')
            return cur.rowcount

    # SQL Execution
    def _execute_sql(self, sql: str | TextClause) -> CursorResult[Any]:
        with self.engine.connect() as conn:
            stmt = text(sql) if isinstance(sql, str) else sql
            cur = conn.execute(stmt)
            self._verbose_print(f'Executed SQL: {stmt}.')
            return cur


class PgsqlDF(Pgsql):
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
        super().__init__(host=host, port=port, user=user, password=password, dbname=dbname, verbose=verbose, **kwargs)

    def df_create_table(
        self,
        df: pd.DataFrame,
        schema_nm: str,
        tb_nm: str,
        primary_col_nm: str = None,
        primary_col_autoinc: Literal['auto', True, False] = 'auto',
        notnull_col_nms: Sequence[str] = None,
        index_col_nms: Sequence[str | Sequence[str]] = None,
        unique_col_nms: Sequence[str | Sequence[str]] = None,
        **kwargs,
    ) -> Table:
        cols = df_to_schema_items(
            df=df,
            tb_nm=tb_nm,
            primary_col_nm=primary_col_nm,
            primary_col_autoinc=primary_col_autoinc,
            notnull_col_nms=notnull_col_nms,
            index_col_nms=index_col_nms,
            unique_col_nms=unique_col_nms,
            dialect='postgresql',
        )
        tb = self.create_table(schema_nm=schema_nm, tb_nm=tb_nm, cols=cols, **kwargs)
        return tb

    def df_add_columns(self, df: pd.DataFrame, schema_nm: str, tb_nm: str, **kwargs) -> Sequence[str]:
        cols = df_to_schema_items(df=df, tb_nm=tb_nm, dialect='postgresql')
        new_col_nms = self.add_columns(schema_nm=schema_nm, tb_nm=tb_nm, cols=cols, **kwargs)
        return new_col_nms

    def df_alter_columns(self, df: pd.DataFrame, schema_nm: str, tb_nm: str, **kwargs) -> Sequence[Column]:
        cols = df_to_schema_items(df=df, tb_nm=tb_nm, dialect='postgresql')
        current_cols = self.get_columns(schema_nm=schema_nm, tb_nm=tb_nm)
        alter_cols = []
        for col in cols:
            if col.name not in current_cols:
                continue
            if isinstance(current_cols[col.name].type, type(col.type)):
                continue
            self.alter_column(schema_nm=schema_nm, tb_nm=tb_nm, col_nm=col.name, sql_dtype=col.type, new_col_nm=None, **kwargs)
            alter_cols.append(col)
        return alter_cols

    def df_copy_rows(self, df: pd.DataFrame, schema_nm: str, tb_nm: str) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        col_nms = map(NameValidator.column, df.columns)
        with BytesIO() as buf:
            df.to_csv(buf, index=False, header=False, sep='\t')
            data = buf.getvalue().replace(b'\r', b'')
        with psycopg2.connect(
            host=self.host, port=self.port, user=self.user, password=self.password, dbname=self.dbname
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO {schema_nm}')
                cur.copy_from(file=BytesIO(data), table=tb_nm, sep='\t', null='', columns=col_nms)
                conn.commit()
                self._verbose_print(f'Copied {df.shape[0]} rows into table {schema_nm}.{tb_nm}.')
                return cur.rowcount

    def df_insert_rows(
        self,
        df: pd.DataFrame,
        schema_nm: str,
        tb_nm: str,
        add_cols: bool = True,
        alter_cols: bool = True,
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        **kwargs,
    ) -> int:
        schema_nm = NameValidator.schema(schema_nm)
        tb_nm = NameValidator.table(tb_nm)
        if add_cols:
            self.df_add_columns(df=df, schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if alter_cols:
            self.df_alter_columns(df=df, schema_nm=schema_nm, tb_nm=tb_nm, **kwargs)
        if on_conflict is None:
            try:
                rowcount = self.df_copy_rows(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
                return rowcount
            except Exception as err:
                self._verbose_print(f'Failed to copy rows: {err}')
        rows = df_to_rows(df=df)
        rowcount = self.insert_rows(
            schema_nm=schema_nm, tb_nm=tb_nm, rows=rows, on_conflict=on_conflict, row_key_validate=False, **kwargs
        )
        return rowcount

    def df_upsert_table(
        self,
        df: pd.DataFrame,
        schema_nm: str,
        tb_nm: str,
        primary_col_nm: str = None,
        notnull_col_nms: Sequence[str] = None,
        index_col_nms: Sequence[str | Sequence[str]] = None,
        unique_col_nms: Sequence[str | Sequence[str]] = None,
    ):
        if not self.table_exists(schema_nm=schema_nm, tb_nm=tb_nm):
            self.df_create_table(
                df=df,
                schema_nm=schema_nm,
                tb_nm=tb_nm,
                primary_col_nm=primary_col_nm,
                notnull_col_nms=notnull_col_nms,
                index_col_nms=index_col_nms,
                unique_col_nms=unique_col_nms,
            )
        else:
            self.df_add_columns(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
            self.df_alter_columns(df=df, schema_nm=schema_nm, tb_nm=tb_nm)
        rowcount = self.df_insert_rows(
            df=df,
            schema_nm=schema_nm,
            tb_nm=tb_nm,
            add_cols=False,
            alter_cols=False,
            on_conflict='upsert',
        )
        return rowcount

    def df_select_rows(
        self,
        schema_nm: str,
        tb_nm: str,
        col_nms: Sequence[str] = None,
        where: Where | Sequence = None,
        order: Order | Sequence[Order] = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> pd.DataFrame:
        col_nms, rows = self.select_rows(
            schema_nm=schema_nm,
            tb_nm=tb_nm,
            col_nms=col_nms,
            where=where,
            order=order,
            limit=limit,
            offset=offset,
            **kwargs,
        )
        df = pd.DataFrame(data=rows, columns=col_nms).convert_dtypes()
        return df

    def df_execute_select(
        self,
        sql: str | TextClause,
    ) -> pd.DataFrame | None:
        cur = self._execute_sql(
            sql=sql,
        )
        try:
            col_nms = list(cur.keys())
            rows = cur.fetchall()
            df = pd.DataFrame(data=rows, columns=col_nms).convert_dtypes()
            return df
        except ResourceClosedError as err:
            self._verbose_print(f'Failed to execute SQL: {err}')
            return
