import os
import pathlib
from typing import Any, Literal, Sequence

import pandas as pd
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
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
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.util import FacadeDict

from .clause import Order, Where, order_parser, where_parser
from .dtype import df_to_rows, df_to_schema_items
from .validator import NameValidator


class Sqlite:
    def __init__(
        self,
        db_path: str | os.PathLike | pathlib.Path = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.db_path = db_path or ':memory:'
        self.verbose = verbose
        self.abs_db_path = self.get_abs_db_path(db_path=self.db_path)
        self.url = self.get_url()
        self.engine = create_engine(
            self.url,
            isolation_level='AUTOCOMMIT',
            connect_args={'check_same_thread': False},
            execution_options={
                'executemany_mode': 'values',
                'executemany_values_page_size': 1000,
            },
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __del__(self) -> None:
        self.engine.dispose()

    def _verbose_print(self, msg: Any):
        if self.verbose:
            print(msg)

    @staticmethod
    def get_abs_db_path(db_path: str | os.PathLike | pathlib.Path) -> str:
        if db_path == ':memory:':
            return db_path
        if str(db_path).startswith(':') and str(db_path).endswith(':'):
            raise ValueError('Only :memory: is allowed for this style of db_path.')
        db_path = os.path.abspath(os.path.normpath(db_path))
        return db_path

    @staticmethod
    def _get_url(abs_db_path: str) -> str:
        url_template = 'sqlite:///{abs_db_path}'
        return url_template.format(abs_db_path=abs_db_path)

    def get_url(self) -> str:
        return self._get_url(self.abs_db_path)

    def validate_conn(self) -> bool:
        with self.engine.connect():
            return True

    def set_wal_journal_mode(self):
        with self.engine.connect() as conn:
            with conn.begin():
                mode = conn.execute(text('PRAGMA journal_mode;')).scalar()
                if mode != 'wal':
                    conn.execute(text('PRAGMA journal_mode = WAL;'))

    def dispose(self):
        self.engine.dispose()

    # Database CRUD
    def create_database(self, db_path: str | os.PathLike | pathlib.Path, **kwargs) -> str:
        if db_path == ':memory:':
            raise ValueError('No need to create a database in memory, just initiate a Sqlite instance.')
        abs_db_path = self.get_abs_db_path(db_path=db_path)
        if os.path.exists(abs_db_path) and os.path.isfile(abs_db_path):
            raise FileExistsError(f'Database already exists at {abs_db_path}.')
        Sqlite(db_path=abs_db_path, **kwargs).validate_conn()
        db = Sqlite(db_path=abs_db_path, **kwargs)
        db.validate_conn()
        db.set_wal_journal_mode()
        self._verbose_print(f'Created database at {abs_db_path}.')
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
        self._verbose_print(f'Dropped database at {abs_db_path}.')
        return abs_db_path

    # Table CRUD
    def create_table(
        self,
        tb_nm: str,
        cols: Sequence[Column | Constraint | Index | UniqueConstraint],
        sqlite_autoincrement: bool = True,
        **kwargs,
    ) -> Table:
        tb_nm = NameValidator.table(tb_nm)
        if self.table_exists(tb_nm=tb_nm):
            raise ValueError(f'Table {tb_nm} already exists.')
        if len(cols) == 0:
            raise ValueError(f'Columns {cols} cannot be empty.')
        for col in cols:
            col.name = NameValidator.column(col.name)
        metadata = MetaData()
        tb = Table(tb_nm, metadata, *cols, sqlite_autoincrement=sqlite_autoincrement)
        tb.create(bind=self.engine, checkfirst=False)
        self._verbose_print(f'Table {tb_nm} created.')
        return self.get_table(tb_nm=tb_nm, **kwargs)

    def table_exists(self, tb_nm: str) -> bool:
        tb_nm = NameValidator.table(tb_nm)
        return inspect(self.engine).has_table(table_name=tb_nm)

    def get_table(self, tb_nm: str, **kwargs) -> Table | None:
        tb_nm = NameValidator.table(tb_nm)
        if not self.table_exists(tb_nm=tb_nm):
            raise ValueError(f'Table {tb_nm} does not exist.')
        metadata = MetaData()
        tb = Table(tb_nm, metadata, autoload_with=self.engine)
        return tb

    def get_tables(self, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        return metadata.tables

    def rename_table(self, tb_nm: str, new_tb_nm: str, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        new_tb_nm = NameValidator.table(new_tb_nm)
        stmt = text(f'ALTER TABLE {tb_nm} RENAME TO {new_tb_nm};')
        self._execute_sql(stmt)
        self._verbose_print(f'Table {tb_nm} renamed to {new_tb_nm}.')
        return new_tb_nm

    def drop_table(self, tb_nm: str, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        tb.drop(bind=self.engine, checkfirst=False)
        self._verbose_print(f'Table {tb_nm} dropped.')
        return tb_nm

    def truncate_table(self, tb_nm: str, restart: bool = True, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        with self.engine.connect() as conn:
            conn.execute(tb.delete())
            if restart:
                if inspect(conn).has_table('sqlite_sequence'):
                    self.delete_rows(tb_nm='sqlite_sequence', where=Where('name', '==', tb_nm))
            self._verbose_print(f'Table {tb_nm} truncated.')
            return tb_nm

    # Column CRUD
    def add_column(self, tb_nm: str, col: Column, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col.name)
        col.name = col_nm
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.add_column(tb_nm, col)
            self._verbose_print(f'Column {col_nm} added to table {tb_nm}.')
            return col_nm

    def add_columns(self, tb_nm: str, cols: Sequence[Column], **kwargs) -> Sequence[str]:
        tb_nm = NameValidator.table(tb_nm)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
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
                    self._verbose_print(f'Column {col_nm} is primary key and cannot be added in table {tb_nm}.')
                    continue
                op.add_column(tb_nm, col)
                new_col_nms.append(col_nm)
            if new_col_nms:
                self._verbose_print(f'Columns {new_col_nms} added to table {tb_nm}.')
            return new_col_nms

    def get_column(self, tb_nm: str, col_nm: str, **kwargs) -> Column:
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        col = tb.columns.get(col_nm)
        if col is None:
            raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
        return col

    def get_columns(self, tb_nm: str, **kwargs) -> dict[str, Column]:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        return {col_nm: col for col_nm, col in tb.columns.items()}

    def rename_column(self, tb_nm: str, col_nm: str, new_col_nm: str, **kwargs) -> Column:
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        new_col_nm = NameValidator.column(new_col_nm)
        if new_col_nm == col_nm:
            raise ValueError(f'New column name {new_col_nm} is same with old column name {col_nm}.')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.alter_column(tb_nm, col_nm, new_column_name=new_col_nm, **kwargs)
            self._verbose_print(f'Column {col_nm} renamed to {new_col_nm} in table {tb_nm}.')
        return self.get_column(tb_nm=tb_nm, col_nm=new_col_nm, **kwargs)

    def alter_column(self, tb_nm: str, col_nm: str, new_col_nm: str, sql_dtype: str = None, **kwargs) -> Column:
        if sql_dtype is not None:
            raise NotImplementedError(
                'SQLite does not support altering a table column datatype directly. You need to create a new table and copy data from original table.'
            )
        return self.rename_column(tb_nm=tb_nm, col_nm=col_nm, new_col_nm=new_col_nm, **kwargs)

    def drop_column(self, tb_nm: str, col_nm: str, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        col_nm = NameValidator.column(col_nm)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_column(tb_nm, col_nm, **kwargs)
            self._verbose_print(f'Dropped column {col_nm} from table {tb_nm}.')
            return col_nm

    def drop_columns(self, tb_nm: str, col_nms: Sequence[str], **kwargs) -> Sequence[str]:
        tb_nm = NameValidator.table(tb_nm)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        dropped_col_nms = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    self._verbose_print(f'Column {col_nm} does not exist in table {tb_nm}.')
                    continue
                op.drop_column(tb_nm, col_nm, **kwargs)
                dropped_col_nms.append(col_nm)
            self._verbose_print(f'Columns {dropped_col_nms} dropped from table {tb_nm}.')
            return dropped_col_nms

    # Index CRUD
    def create_index(self, tb_nm: str, col_nms: Sequence[str], idx_nm: str = None, unique: bool = False, **kwargs) -> Index:
        tb_nm = NameValidator.table(tb_nm)
        col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        index_col_nms = []
        for col_nm in col_nms:
            if col_nm not in current_cols:
                raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
            if col_nm in index_col_nms:
                raise ValueError(f'Column {col_nm} is duplicated in {col_nms}.')
            index_col_nms.append(col_nm)
        idx_nm = NameValidator.index(idx_nm or f'idx_{tb_nm}_{"_".join(index_col_nms)}')
        if idx_nm in self.get_indexes(tb_nm=tb_nm, **kwargs):
            raise ValueError(f'Index {idx_nm} already exists.')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.create_index(idx_nm, tb_nm, col_nms, unique=unique, **kwargs)
            self._verbose_print(f'Index {idx_nm} created on table {tb_nm} with columns {col_nms}.')
        return self.get_index(tb_nm=tb_nm, idx_nm=idx_nm, **kwargs)

    def get_index(self, tb_nm: str, col_nms: Sequence[str] = None, idx_nm: str = None, **kwargs) -> Index:
        if idx_nm is None and col_nms is None:
            raise ValueError('Either idx_nm or col_nms must be provided.')
        indexes = self.get_indexes(tb_nm=tb_nm, **kwargs)
        if idx_nm is not None:
            if idx_nm in indexes:
                return indexes.get(idx_nm)
            raise ValueError(f'Index {idx_nm} does not exist in table {tb_nm}.')
        if col_nms is not None:
            col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
            current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
            for col_nm in col_nms:
                if col_nm not in current_cols:
                    raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
            for idx_nm, idx in indexes.items():
                if set(idx.columns.keys()) == set(col_nms):
                    return idx
            raise ValueError(f'Index with columns {col_nms} does not exist in table {tb_nm}.')

    def get_indexes(self, tb_nm: str, **kwargs) -> dict[str, Index]:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        return {idx.name: idx for idx in tb.indexes}

    def drop_index(self, tb_nm: str, col_nms: Sequence[str] = None, idx_nm: str = None, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        idx = self.get_index(tb_nm=tb_nm, col_nms=col_nms, idx_nm=idx_nm, **kwargs)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_index(idx.name, tb_nm, if_exists=False, **kwargs)
            self._verbose_print(f'Index {idx.name} dropped from table {tb_nm}.')
            return idx.name

    # Rows CRUD
    def insert_row(
        self,
        tb_nm: str,
        row: dict[str, Any],
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        if row_key_validate:
            row = {NameValidator.column(key): val for key, val in row.items()}
        with self.engine.connect() as conn:
            if on_conflict in ('update', 'replace', 'upsert'):
                with conn.begin():
                    cur = conn.execute(tb.insert().prefix_with('OR REPLACE').values(**row))
                self._verbose_print(f'Inserted or replaced row into table {tb_nm}.')
            elif on_conflict in ('do_nothing', 'ignore', 'skip'):
                with conn.begin():
                    cur = conn.execute(insert(tb).values(**row).on_conflict_do_nothing())
                self._verbose_print(f'Inserted or ignored row into table {tb_nm}.')
            elif on_conflict is None:
                with conn.begin():
                    cur = conn.execute(tb.insert(), row)
                self._verbose_print(f'Inserted row into table {tb_nm}.')
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}.')
            return cur.rowcount

    def insert_rows(
        self,
        tb_nm: str,
        rows: Sequence[dict[str, Any]],
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
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
                self._verbose_print(f'Inserted or replaced {len(rows)} rows into table {tb_nm}.')
            elif on_conflict == 'do_nothing':
                stmt = insert(tb).on_conflict_do_nothing()
                with conn.begin():
                    cur = conn.execute(stmt, rows)
                self._verbose_print(f'Inserted or do nothing {len(rows)} rows into table {tb_nm}.')
            elif on_conflict in ('ignore', 'skip'):
                stmt = insert(tb).prefix_with('OR IGNORE')
                with conn.begin():
                    cur = conn.execute(stmt, rows)
                self._verbose_print(f'Inserted or ignored {len(rows)} rows into table {tb_nm}.')
            elif on_conflict is None:
                stmt = insert(tb)
                with conn.begin():
                    conn.execute(text('PRAGMA journal_mode = OFF;'))
                    conn.execute(text('PRAGMA synchronous = OFF;'))
                    conn.execute(text('PRAGMA temp_store = MEMORY;'))
                    cur = conn.execute(stmt, rows)
                    conn.execute(text('PRAGMA journal_mode = WAL;'))
                    conn.execute(text('PRAGMA synchronous = NORMAL;'))
                    conn.execute(text('PRAGMA temp_store = DEFAULT;'))
                self._verbose_print(f'Inserted rows into table {tb_nm}.')
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}.')
            return cur.rowcount

    def select_rows(
        self,
        tb_nm: str,
        col_nms: Sequence[str] = None,
        where: Where | Sequence = None,
        order: Order | Sequence[Order] = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> tuple[list[str], list[Row]] | None:
        tb_nm = NameValidator.table(tb_nm)
        # self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        if col_nms is None:
            selected_cols = list(current_cols.values())
        else:
            selected_cols = []
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
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
            self._verbose_print(f'Selected {len(rows)} rows from table {tb_nm}.')
            return cols, rows

    def update_rows(self, tb_nm: str, set_values: dict[str, Any], where: Where | Sequence = None, **kwargs) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        _set_values = {}
        for col_nm, val in set_values.items():
            col_nm = NameValidator.column(col_nm)
            if col_nm not in tb.columns:
                raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}.')
            _set_values[col_nm] = val
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.update().where(where_cond)
            with conn.begin():
                cur = conn.execute(stmt, _set_values)
            self._verbose_print(f'Updated {cur.rowcount} rows in table {tb_nm}.')
            return cur.rowcount

    def delete_rows(self, tb_nm: str, where: Where | Sequence = None, **kwargs) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.delete().where(where_cond)
            with conn.begin():
                cur = conn.execute(stmt)
            self._verbose_print(f'Deleted {cur.rowcount} rows from table {tb_nm}.')
            return cur.rowcount

    # SQL Execution
    def _execute_sql(self, sql: str | TextClause) -> CursorResult[Any]:
        with self.engine.connect() as conn:
            stmt = text(sql) if isinstance(sql, str) else sql
            cur = conn.execute(stmt)
            self._verbose_print(f'Executed SQL: {stmt}.')
            return cur


class SqliteDF(Sqlite):
    def __init__(self, db_path: str | os.PathLike | pathlib.Path = None, verbose: bool = False, **kwargs):
        super().__init__(db_path=db_path, verbose=verbose, **kwargs)

    def df_create_table(
        self,
        df: pd.DataFrame,
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
            dialect='sqlite',
        )
        tb = self.create_table(tb_nm=tb_nm, cols=cols, **kwargs)
        return tb

    def df_add_columns(self, df: pd.DataFrame, tb_nm: str, **kwargs) -> Sequence[str]:
        cols = df_to_schema_items(df=df, tb_nm=tb_nm, dialect='sqlite')
        new_col_nms = self.add_columns(tb_nm=tb_nm, cols=cols, **kwargs)
        return new_col_nms

    def df_insert_rows(
        self,
        df: pd.DataFrame,
        tb_nm: str,
        add_cols: bool = True,
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        **kwargs,
    ) -> int:
        tb_nm = NameValidator.table(tb_nm)
        if add_cols:
            self.df_add_columns(df=df, tb_nm=tb_nm, **kwargs)
        rows = df_to_rows(df=df, dialect='sqlite')
        rowcount = self.insert_rows(tb_nm=tb_nm, rows=rows, on_conflict=on_conflict, row_key_validate=False, **kwargs)
        return rowcount

    def df_upsert_table(
        self,
        df: pd.DataFrame,
        tb_nm: str,
        primary_col_nm: str = None,
        primary_col_autoinc: Literal['auto', True, False] = 'auto',
        notnull_col_nms: Sequence[str] = None,
        index_col_nms: Sequence[str | Sequence[str]] = None,
        unique_col_nms: Sequence[str | Sequence[str]] = None,
    ):
        if not self.table_exists(tb_nm=tb_nm):
            self.df_create_table(
                df=df,
                tb_nm=tb_nm,
                primary_col_nm=primary_col_nm,
                primary_col_autoinc=primary_col_autoinc,
                notnull_col_nms=notnull_col_nms,
                index_col_nms=index_col_nms,
                unique_col_nms=unique_col_nms,
            )
        else:
            self.df_add_columns(df=df, tb_nm=tb_nm)
        rowcount = self.df_insert_rows(
            df=df,
            tb_nm=tb_nm,
            add_cols=False,
            on_conflict='upsert',
        )
        return rowcount

    def df_select_rows(
        self,
        tb_nm: str,
        col_nms: Sequence[str] = None,
        where: Where | Sequence = None,
        order: Order | Sequence[Order] = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> pd.DataFrame:
        col_nms, rows = self.select_rows(
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
