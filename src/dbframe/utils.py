import re
from typing import Literal, NamedTuple, Sequence

from sqlalchemy import and_, or_, Table, true, Column

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
        raise ValueError(f'Where clause must have one of columns or table argument')

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
