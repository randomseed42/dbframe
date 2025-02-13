import re
from typing import Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import and_, Boolean, Column, DateTime, Float, Integer, or_, String, Table, Text, Time, true
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
    bool: Boolean,
    np.bool: Boolean,
    pd.BooleanDtype(): Boolean,

    # Integer Types
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

    # Float Types
    float: Float,
    np.float16: Float,
    np.float32: Float,
    np.float64: Float,
    pd.Float32Dtype(): Float,
    pd.Float64Dtype(): Float,

    # String Types
    str: String,
    np.str_: String,
    np.object_: String,
    pd.StringDtype(): String,
    pd.CategoricalDtype.type: String,

    # DateTime Types
    np.datetime64: DateTime,
    pd.DatetimeTZDtype: DateTime(timezone=True),
    pd.DatetimeIndex: DateTime,
    pd.Timestamp: DateTime,

    # Time Types
    np.timedelta64: Time,
    pd.Timedelta: Time,

    # pd.Period: Date,
    # pd.PeriodDtype('Y'): Date,
    # pd.PeriodDtype('M'): Date,
    # pd.PeriodDtype('W'): Date,
    # pd.PeriodDtype('W-MON'): Date,
    # pd.PeriodDtype('W-TUE'): Date,
    # pd.PeriodDtype('W-WED'): Date,
    # pd.PeriodDtype('W-THU'): Date,
    # pd.PeriodDtype('W-FRI'): Date,
    # pd.PeriodDtype('W-SAT'): Date,
    # pd.PeriodDtype('W-SUN'): Date,
    # pd.PeriodDtype('D'): Date,
    # pd.PeriodDtype('h'): DateTime,
    # pd.PeriodDtype('min'): DateTime,
    # pd.PeriodDtype('s'): DateTime,
    # pd.PeriodDtype('ms'): DateTime,
    # pd.PeriodDtype('us'): DateTime,
    # pd.PeriodDtype('ns'): DateTime,
    # pd.DateOffset: Date,

    # Text for large strings
    # pd.ArrowDtype(): Text,
}


def series_to_sql_dtype(s: np.typing.NDArray | pd.Series | pd.DatetimeIndex | pd.Categorical) -> TypeEngine:
    if hasattr(s.dtype, 'subtype'):
        python_type = s.dtype.subtype
    elif hasattr(s.dtype, 'freq'):
        python_type = s.dtype
    elif hasattr(s.dtype, 'tz'):
        python_type = pd.DatetimeTZDtype
    else:
        python_type = s.dtype.type
    sql_dtype = SQL_DTYPE_MAP.get(python_type, Text)
    return sql_dtype


def dataframe_to_columns(
        df: pd.DataFrame,
        primary_col: str = None,
        unique_cols: list[str] = None,
        notnull_cols: list[str] = None,
        index_cols: list[str | list[str]] = None,
) -> list[Column]:
    ...