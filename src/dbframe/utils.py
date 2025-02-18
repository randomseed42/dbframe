import re
from datetime import datetime, time, timedelta
from typing import Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Boolean, Column, Constraint, DateTime, Float, Index, Integer, String, Table, Text, Time, \
    UniqueConstraint, and_, or_, true
from sqlalchemy.sql.sqltypes import TypeEngine

WhereOperator = Literal[
    '==',
    '!=',
    '<',
    '<=',
    '>',
    '>=',
    'in',
    'not in',
    'like',
    'not like',
    'ilike',
    'not ilike',
    'between',
    'is',
    'is not',
]

OPERATORS_MAP = {
    '==': '__eq__',
    '!=': '__ne__',
    '>': '__gt__',
    '>=': '__ge__',
    '<': '__lt__',
    '<=': '__le__',
    'in': 'in_',
    'not in': 'not_in',
    'like': 'like',
    'not like': 'not_like',
    'ilike': 'ilike',
    'not ilike': 'not_ilike',
    'between': 'between',
    'is': 'is_',
    'is_not': 'is_not',
}


class WhereClause(NamedTuple):
    column_name: str
    operator: WhereOperator
    value: str | int | float | bool | Sequence | None


def where_clauses_parser(
        where_clauses: list | tuple | WhereClause | None,
        columns: dict[str, Column] = None,
        table: Table = None,
):
    if not (columns is None or len(columns) == 0) ^ (table is None):
        raise ValueError('Where clause must have one of `columns` or `table` argument')

    def _where_parser(_where: WhereClause | None):
        _columns = columns or table.columns
        if _where is None:
            return true()
        column_name = NamingValidator.column(_where.column_name)
        if column_name not in _columns:
            raise ValueError(f'Where clause column {column_name} does not exist in table')
        if _where.operator == 'between':
            if not isinstance(_where.value, Sequence) or len(_where.value) != 2:
                raise ValueError(f'Expected two items in between clause, got {_where.value}')
            cond = getattr(_columns.get(column_name), OPERATORS_MAP[_where.operator])(*_where.value)
            return cond
        if _where.operator != 'between':
            cond = getattr(_columns.get(column_name), OPERATORS_MAP[_where.operator])(_where.value)
            return cond

    if where_clauses is None:
        return true()
    if isinstance(where_clauses, WhereClause):
        return _where_parser(where_clauses)
    if isinstance(where_clauses, list):
        return and_(
            where_clauses_parser(where_clauses=_where, columns=columns, table=table)
            for _where in where_clauses
        )
    if isinstance(where_clauses, tuple):
        return or_(
            where_clauses_parser(where_clauses=_where, columns=columns, table=table)
            for _where in where_clauses
        )


class OrderByClause(NamedTuple):
    column_name: str
    ascending: bool = True


def order_by_parser(order_by: list[OrderByClause] | None, columns: dict[str, Column]):
    if order_by is None:
        return [None]
    orders = []
    for order in order_by:
        column_name = NamingValidator.column(order.column_name)
        if order.ascending:
            orders.append(columns.get(column_name).asc())
        else:
            orders.append(columns.get(order.column_name).desc())
    return orders


class NamingValidator:
    @classmethod
    def dbname(cls, name: str) -> str:
        if not re.match('^[a-zA-Z][a-zA-Z0-9_]*$', name):
            raise ValueError(f'The dbname {name} is invalid')
        return name.lower()

    @classmethod
    def schema(cls, name: str) -> str:
        if not re.match('^[a-zA-Z][a-zA-Z0-9_]*$', name):
            raise ValueError(f'The schema name {name} is invalid')
        return name.lower()

    @classmethod
    def table(cls, name: str) -> str:
        if not re.match('^[a-zA-Z][a-zA-Z0-9_]*$', name):
            raise ValueError(f'The table name {name} is invalid')
        return name.lower()

    @classmethod
    def column(cls, name: str) -> str:
        if not re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f'The column name {name} is invalid')
        return name.lower()

"""
Pandas dtypes include @register_extension_dtype
"""
SQL_DTYPE_MAP = {
    # Bool Types
    'bool': Boolean,
    bool: Boolean,
    np.bool: Boolean,
    pd.BooleanDtype(): Boolean,

    # Integer Types
    'int': Integer,
    int: Integer,
    np.int8: Integer,
    np.int16: Integer,
    np.int32: Integer,
    np.int64: Integer,
    pd.Int8Dtype(): Integer,
    pd.Int16Dtype(): Integer,
    pd.Int32Dtype(): Integer,
    pd.Int64Dtype(): Integer,
    pd.UInt8Dtype(): Integer,
    pd.UInt16Dtype(): Integer,
    pd.UInt32Dtype(): Integer,
    pd.UInt64Dtype(): Integer,
    pd.RangeIndex: Integer,

    # Float Types
    'float': Float,
    float: Float,
    np.float16: Float,
    np.float32: Float,
    np.float64: Float,
    pd.Float32Dtype(): Float,
    pd.Float64Dtype(): Float,

    # String Types
    'str': String,
    str: String,
    np.str_: String,
    np.object_: String,
    pd.StringDtype(): String,
    pd.CategoricalDtype.type: String,

    # DateTime Types
    'datetime': DateTime,
    datetime: DateTime,
    np.datetime64: DateTime,
    pd.DatetimeTZDtype: DateTime(timezone=True),
    pd.DatetimeIndex: DateTime,
    pd.Timestamp: DateTime,

    # Time Types
    'time': Time,
    time: Time,
    timedelta: Time,
    np.timedelta64: Time,
    pd.Timedelta: Time,
}


def series_to_sql_dtype(s: np.typing.NDArray | pd.Series | pd.DatetimeIndex | pd.RangeIndex | pd.MultiIndex | pd.Index | pd.Categorical, datetime_fmt: str = '%Y-%m-%d %H:%M:%S', **kwargs) -> TypeEngine:
    if hasattr(s.dtype, 'subtype'):
        python_type = s.dtype.subtype
    elif hasattr(s.dtype, 'freq'):
        python_type = s.dtype
    elif hasattr(s.dtype, 'tz'):
        python_type = pd.DatetimeTZDtype
    elif s.dtype.type in [
        'str',
        str,
        np.str_,
        np.object_,
        pd.StringDtype(),
    ]:
        if len(s) == 0:
            python_type = s.dtype.type
        else:
            try:
                pd.to_datetime(s, errors='raise', format=datetime_fmt, **kwargs)
                python_type = datetime
            except ValueError:
                python_type = s.dtype.type
    else:
        python_type = s.dtype.type
    sql_dtype = SQL_DTYPE_MAP.get(python_type, Text)
    return sql_dtype


def _df_to_sql_primary_column(df: pd.DataFrame, df_column_name: str | Literal['index'] = None, sql_column_name: str = None, **kwargs) -> Column | None:
    if df_column_name is None:
        return None
    if df_column_name == 'index':
        column_name = sql_column_name or 'uid'
        column_name = NamingValidator.column(column_name)
        sql_dtype = series_to_sql_dtype(df.index, **kwargs)
        return Column(column_name, sql_dtype, primary_key=True, nullable=False, autoincrement='auto')
    if df_column_name not in df.columns:
        raise ValueError(f'Column {df_column_name} does not exist in df')
    column_name = sql_column_name or str(df_column_name)
    column_name = NamingValidator.column(column_name)
    sql_dtype = series_to_sql_dtype(df[df_column_name], **kwargs)
    return Column(column_name, sql_dtype, primary_key=True, nullable=False, autoincrement='auto')


def _df_to_sql_index(df: pd.DataFrame, df_column_names: str | list[str], table_name: str) -> Index | None:
    if isinstance(df_column_names, str):
        df_column_names = [df_column_names]
    index_column_names = []
    for df_column_name in df_column_names:
        if df_column_name not in df.columns:
            raise ValueError(f'Column {df_column_name} does not exist in df')
        column_name = NamingValidator.column(df_column_name)
        if column_name in index_column_names:
            raise ValueError(f'Column {df_column_name} is duplicated')
        index_column_names.append(column_name)
    idx_name = 'ix_{}_{}'.format(table_name, '_'.join(index_column_names))
    index = Index(idx_name, *index_column_names)
    return index


def _df_to_sql_unique(df: pd.DataFrame, df_column_names: str | list[str], table_name: str) -> UniqueConstraint | None:
    if isinstance(df_column_names, str):
        df_column_names = [df_column_names]
    unique_column_names = []
    for df_column_name in df_column_names:
        if df_column_name not in df.columns:
            raise ValueError(f'Column {df_column_name} does not exist in df')
        column_name = NamingValidator.column(df_column_name)
        if column_name in unique_column_names:
            raise ValueError(f'Column {df_column_name} is duplicated')
        unique_column_names.append(column_name)
    idx_name = 'uix_{}_{}'.format(table_name, '_'.join(unique_column_names))
    unique_constraint = UniqueConstraint(*unique_column_names, name=idx_name)
    return unique_constraint


def df_to_sql_columns(
        df: pd.DataFrame,
        table_name: str,
        primary_column_name: str | Literal['index'] = None,
        primary_sql_column_name: str = None,
        notnull_column_names: list[str] = None,
        index_column_names: list[str | list[str]] = None,
        unique_column_names: list[str | list[str]] = None,
        **kwargs
) -> list[Column | Constraint | Index] | None:
    table_name = NamingValidator.table(table_name)
    schema_items = []
    primary_column = _df_to_sql_primary_column(df=df, df_column_name=primary_column_name, sql_column_name=primary_sql_column_name)
    if primary_column is not None:
        schema_items.append(primary_column)
    if notnull_column_names is None:
        notnull_column_names = []
    if index_column_names is None:
        index_column_names = []
    if unique_column_names is None:
        unique_column_names = []
    for df_column_name in df.columns:
        if df_column_name == primary_column_name:
            continue
        column_name = NamingValidator.column(df_column_name)
        sql_dtype = series_to_sql_dtype(df[df_column_name], **kwargs)
        is_nullable = df_column_name not in notnull_column_names
        schema_items.append(Column(column_name, sql_dtype, nullable=is_nullable))
    for index_column_names_group in index_column_names:
        composite_index = _df_to_sql_index(df=df, df_column_names=index_column_names_group, table_name=table_name)
        schema_items.append(composite_index)
    for unique_column_names_group in unique_column_names:
        composite_unique = _df_to_sql_unique(df=df, df_column_names=unique_column_names_group, table_name=table_name)
        schema_items.append(composite_unique)
    return schema_items
