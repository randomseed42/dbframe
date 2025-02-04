from typing import Literal, NamedTuple, Sequence

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


class OrderByClause(NamedTuple):
    column_name: str
    ascending: bool = True
