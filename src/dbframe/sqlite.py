import os
import pathlib
from typing import Any, Literal, Sequence

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, CursorResult, Index, MetaData, Row, Table, create_engine, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.util import FacadeDict

from .clause import Order, Where, order_parser, where_parser
from .validator import NameValidator


class Sqlite:
    def __init__(self, db_path: str | os.PathLike | pathlib.Path = None, verbose: bool = False, **kwargs):
        self.db_path = db_path or ':memory:'
        self.verbose = verbose
        self.abs_db_path = self.get_abs_db_path(db_path=self.db_path)
        self.url = self.get_url()
        self.engine = create_engine(self.url, isolation_level='AUTOCOMMIT', connect_args={'check_same_thread': False})

    def __del__(self) -> None:
        self.engine.dispose()

    def _verbose_print(self, msg: Any):
        if self.verbose:
            print(msg)

    @staticmethod
    def get_abs_db_path(db_path: str | os.PathLike | pathlib.Path) -> str:
        if db_path == ':memory:':
            return db_path
        db_path = os.path.abspath(os.path.normpath(db_path))
        return db_path

    def get_url(self) -> str:
        return f'sqlite:///{self.abs_db_path}'

    def validate_conn(self) -> bool:
        with self.engine.connect():
            return True

    # Database CRUD
    def create_database(self, db_path: str | os.PathLike | pathlib.Path, **kwargs) -> str:
        if db_path == ':memory:':
            raise ValueError('No need to create a database in memory, just initiate a Sqlite instance.')
        abs_db_path = self.get_abs_db_path(db_path=db_path)
        if os.path.exists(abs_db_path) and os.path.isfile(abs_db_path):
            raise FileExistsError(f'Database already exists at {abs_db_path}')
        Sqlite(db_path=abs_db_path, **kwargs).validate_conn()
        self._verbose_print(f'Created database at {abs_db_path}')
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
        self._verbose_print(f'Dropped database at {abs_db_path}')
        return abs_db_path

    # Table CRUD
    def get_table(self, tb_nm: str, **kwargs) -> Table | None:
        tb_nm = NameValidator.table(tb_nm)
        metadata = MetaData()
        try:
            tb = Table(tb_nm, metadata, autoload_with=self.engine, **kwargs)
            return tb
        except NoSuchTableError:
            raise ValueError(f'Table {tb_nm} does not exist')

    def get_tables(self, views: bool = False, **kwargs) -> dict[str, Table] | FacadeDict | None:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, views=views, **kwargs)
        return metadata.tables

    def create_table(self, tb_nm: str, cols: list[Column], **kwargs) -> Table | None:
        tb_nm = NameValidator.table(tb_nm)
        try:
            self.get_table(tb_nm=tb_nm)
            raise ValueError(f'Table {tb_nm} already exists')
        except ValueError:
            pass
        if len(cols) == 0:
            raise ValueError(f'Columns {cols} cannot be empty')
        metadata = MetaData()
        for col in cols:
            col.name = NameValidator.column(col.name)
        tb = Table(tb_nm, metadata, *cols, **kwargs)
        tb.create(bind=self.engine, checkfirst=True)
        self._verbose_print(f'Table {tb_nm} created')
        return self.get_table(tb_nm=tb_nm, **kwargs)

    def rename_table(self, tb_nm: str, new_tb_nm: str, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        new_tb_nm = NameValidator.table(new_tb_nm)
        try:
            self.get_table(tb_nm=new_tb_nm, **kwargs)
            raise ValueError(f'New table {new_tb_nm} already exists')
        except ValueError:
            pass
        with self.engine.connect() as conn:
            stmt = text(f'ALTER TABLE {tb_nm} RENAME TO {new_tb_nm};')
            conn.execute(stmt)
            self._verbose_print(f'Old table {tb_nm} has been renamed to {new_tb_nm}')
            return new_tb_nm

    def drop_table(self, tb_nm: str, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        tb.drop(bind=self.engine, checkfirst=True)
        self._verbose_print(f'Table {tb_nm} dropped')
        return tb_nm

    def truncate_table(self, tb_nm: str, restart: bool = True, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        with self.engine.connect() as conn:
            conn.execute(text(f'DELETE FROM {tb_nm};'))
            if restart:
                if self.get_table(tb_nm='sqlite_sequence') is not None:
                    conn.execute(text(f"DELETE FROM sqlite_sequence WHERE name='{tb_nm}';"))
            self._verbose_print(f'Table {tb_nm} truncated')
            return tb_nm

    # Column CRUD
    def add_column(self, tb_nm: str, col: Column, **kwargs) -> str | None:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        col_nm = NameValidator.column(col.name)
        col.name = col_nm
        try:
            self.get_column(tb_nm=tb_nm, col_nm=col_nm, **kwargs)
            raise ValueError(f'Column {col_nm} already exists in table {tb_nm}')
        except ValueError:
            pass
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.add_column(tb_nm, col)
            self._verbose_print(f'Added column {col_nm} to table {tb_nm}')
            return col_nm

    def add_columns(self, tb_nm: str, cols: list[Column], **kwargs) -> list[str] | None:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        new_col_nms = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for col in cols:
                col_nm = NameValidator.column(col.name)
                col.name = col_nm
                if col_nm in current_cols or col_nm in new_col_nms:
                    self._verbose_print(f'Column {col_nm} already exists in table {tb_nm}')
                    continue
                op.add_column(tb_nm, col)
                new_col_nms.append(col_nm)
            self._verbose_print(f'Added columns {new_col_nms} to table {tb_nm}')
            return new_col_nms

    def get_column(self, tb_nm: str, col_nm: str, **kwargs) -> Column:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        col_nm = NameValidator.column(col_nm)
        col = tb.columns.get(col_nm)
        if col is None:
            raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}')
        return col

    def get_columns(self, tb_nm: str, **kwargs) -> dict[str, Column] | None:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        return {col_nm: col for col_nm, col in tb.columns.items()}

    def rename_column(self, tb_nm: str, col_nm: str, new_col_nm: str, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        col_nm = NameValidator.column(col_nm)
        self.get_column(tb_nm=tb_nm, col_nm=col_nm, **kwargs)
        new_col_nm = NameValidator.column(new_col_nm)
        if new_col_nm == col_nm:
            raise ValueError(f'New column name {new_col_nm} is same with old column name {col_nm}')
        try:
            self.get_column(tb_nm=tb_nm, col_nm=new_col_nm, **kwargs)
            raise ValueError(f'New column name {new_col_nm} already exists in table {tb_nm}')
        except ValueError:
            pass
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.alter_column(tb_nm, col_nm, new_column_name=new_col_nm, **kwargs)
            self._verbose_print(f'Renamed column {col_nm} to {new_col_nm} in table {tb_nm}')
            return new_col_nm

    def alter_column(self, tb_nm: str, col_nm: str, new_col_nm: str, sql_dtype: str = None, **kwargs) -> str:
        if sql_dtype is not None:
            raise NotImplementedError(
                'SQLite does not support altering a table column datatype directly. You need to create a new table and copy data from original table'
            )
        return self.rename_column(tb_nm=tb_nm, col_nm=col_nm, new_col_nm=new_col_nm, **kwargs)

    def drop_column(self, tb_nm: str, col_nm: str, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        col_nm = NameValidator.column(col_nm)
        self.get_column(tb_nm=tb_nm, col_nm=col_nm, **kwargs)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_column(tb_nm, col_nm, **kwargs)
            self._verbose_print(f'Dropped column {col_nm} from table {tb_nm}')
            return col_nm

    def drop_columns(self, tb_nm: str, col_nms: list[str], **kwargs) -> list[str]:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        dropped_col_nms = []
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    self._verbose_print(f'Column {col_nm} does not exist in table {tb_nm}')
                    continue
                op.drop_column(tb_nm, col_nm, **kwargs)
                dropped_col_nms.append(col_nm)
            self._verbose_print(f'Dropped columns {dropped_col_nms} from table {tb_nm}')
            return dropped_col_nms

    # Index CRUD
    def create_index(self, tb_nm: str, col_nms: list[str], idx_nm: str = None, unique: bool = False, **kwargs) -> str:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
        index_col_nms = []
        for col_nm in col_nms:
            if col_nm not in current_cols:
                raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}')
            if col_nm in index_col_nms:
                raise ValueError(f'Column {col_nm} is duplicated in {col_nms}')
            index_col_nms.append(col_nm)
        if idx_nm is None:
            idx_nm = f'idx_{tb_nm}_{"_".join(index_col_nms)}'
        if idx_nm in self.get_indexes(tb_nm=tb_nm, **kwargs):
            raise ValueError(f'Index {idx_nm} already exists')
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.create_index(idx_nm, tb_nm, col_nms, unique=unique, **kwargs)
            self._verbose_print(f'Created index {idx_nm} on table {tb_nm} with columns {col_nms}')
            return idx_nm

    def get_index(self, tb_nm: str, col_nms: list[str] = None, idx_nm: str = None, **kwargs) -> Index:
        indexes = self.get_indexes(tb_nm=tb_nm, **kwargs)
        if idx_nm is None and col_nms is None:
            raise ValueError('Either idx_nm or col_nms must be provided')
        if idx_nm is not None:
            if idx_nm in indexes:
                return indexes.get(idx_nm)
            raise ValueError(f'Index {idx_nm} does not exist in table {tb_nm}')
        if col_nms is not None:
            col_nms = [NameValidator.column(col_nm) for col_nm in col_nms]
            for col_nm in col_nms:
                if col_nm not in self.get_columns(tb_nm=tb_nm, **kwargs):
                    raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}')
            for idx in indexes:
                if set(idx.columns.keys()) == set(col_nms):
                    return idx
            raise ValueError(f'Index with columns {col_nms} does not exist in table {tb_nm}')

    def get_indexes(self, tb_nm: str, **kwargs) -> dict[str, Index]:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        return {idx.name: idx for idx in tb.indexes}

    def drop_index(self, tb_nm: str, col_nms: list[str] = None, idx_nm: str = None, **kwargs) -> str:
        idx = self.get_index(tb_nm=tb_nm, col_nms=col_nms, idx_nm=idx_nm, **kwargs)
        with self.engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            op.drop_index(idx.name, tb_nm, if_exists=True, **kwargs)
            self._verbose_print(f'Dropped index {idx.name} from table {tb_nm}')
            return idx.name

    # Rows CRUD
    def insert_row(
        self,
        tb_nm: str,
        row: dict,
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> Row:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        if row_key_validate:
            row = {NameValidator.column(key): val for key, val in row.items()}
        with self.engine.connect() as conn:
            if on_conflict in ('update', 'replace', 'upsert'):
                cur = conn.execute(tb.insert().prefix_with('OR REPLACE').values(**row))
                self._verbose_print(f'Inserted or replaced row into table {tb_nm}')
            elif on_conflict in ('do_nothing', 'ignore', 'skip'):
                # cur = conn.execute(tb.insert().prefix_with('OR IGNORE').values(**row))
                cur = conn.execute(insert(tb).values(**row).on_conflict_do_nothing())
                self._verbose_print(f'Inserted or ignored row into table {tb_nm}')
            elif on_conflict is None:
                cur = conn.execute(tb.insert(), row)
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}')
            return cur.inserted_primary_key

    def insert_rows(
        self,
        tb_nm: str,
        rows: Sequence[dict],
        on_conflict: Literal['do_nothing', 'ignore', 'skip', 'update', 'replace', 'upsert'] = None,
        row_key_validate: bool = False,
        **kwargs,
    ) -> list[Row]:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        if row_key_validate:
            rows = [{NameValidator.column(key): val for key, val in row.items()} for row in rows]
        with self.engine.connect() as conn:
            if on_conflict in ('update', 'replace', 'upsert'):
                cur = conn.execute(tb.insert().prefix_with('OR REPLACE'), rows)
                self._verbose_print(f'Inserted or replaced {len(rows)} rows into table {tb_nm}')
            elif on_conflict == 'do_nothing':
                cur = conn.execute(insert(tb).values(rows).on_conflict_do_nothing())
                self._verbose_print(f'Inserted or do nothing {len(rows)} rows into table {tb_nm}')
            elif on_conflict in ('ignore', 'skip'):
                cur = conn.execute(tb.insert().prefix_with('OR IGNORE'), rows)
                self._verbose_print(f'Inserted or ignored {len(rows)} rows into table {tb_nm}')
            elif on_conflict is None:
                cur = conn.execute(tb.insert(), rows)
            else:
                raise ValueError(f'Invalid on_conflict value: {on_conflict}')
            return cur.inserted_primary_key_rows

    def select_rows(
        self,
        tb_nm: str,
        col_nms: list[str] = None,
        where: Where | Sequence = None,
        order: Order | Sequence[Order] = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> tuple[list[str], list[Row]] | None:
        tb_nm = NameValidator.table(tb_nm)
        self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        if col_nms is None:
            selected_cols = list(current_cols.values())
        else:
            selected_cols = []
            for col_nm in col_nms:
                col_nm = NameValidator.column(col_nm)
                if col_nm not in current_cols:
                    raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}')
                selected_cols.append(current_cols[col_nm])
        if len(selected_cols) == 0:
            raise ValueError('Columns cannot be empty')

        where_cond = where_parser(where, cols=current_cols, tb=None)
        order_cond = order_parser(order, cols=current_cols, tb=None)
        with self.engine.connect() as conn:
            stmt = select(*selected_cols).where(where_cond).order_by(*order_cond).offset(offset).limit(limit)
            cur = conn.execute(stmt)
            cols = list(cur.keys())
            rows = cur.fetchall()
            self._verbose_print(f'Selected {len(rows)} rows from table {tb_nm}')
            return cols, rows

    def update_rows(self, tb_nm: str, set_values: dict, where: Where | Sequence = None, **kwargs) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        current_cols = self.get_columns(tb_nm=tb_nm, **kwargs)
        _set_values = {}
        for col_nm, val in set_values.items():
            col_nm = NameValidator.column(col_nm)
            if col_nm not in current_cols:
                raise ValueError(f'Column {col_nm} does not exist in table {tb_nm}')
            _set_values[col_nm] = val
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.update().where(where_cond)
            cur = conn.execute(stmt, _set_values)
            self._verbose_print(f'Updated {cur.rowcount} rows in table {tb_nm}')
            return cur.rowcount

    def delete_rows(self, tb_nm: str, where: Where | Sequence = None, **kwargs) -> int:
        tb_nm = NameValidator.table(tb_nm)
        tb = self.get_table(tb_nm=tb_nm, **kwargs)
        where_cond = where_parser(where, cols=None, tb=tb)
        with self.engine.connect() as conn:
            stmt = tb.delete().where(where_cond)
            cur = conn.execute(stmt)
            self._verbose_print(f'Deleted {cur.rowcount} rows from table {tb_nm}')
            return cur.rowcount

    # Execute SQL
    def _execute_sql(self, sql: str) -> CursorResult[Any]:
        with self.engine.connect() as conn:
            cur = conn.execute(text(sql))
            self._verbose_print(f'Executed SQL: {sql}')
            return cur


class SqliteDF(Sqlite):
    def __init__(self):
        super().__init__()
        pass
