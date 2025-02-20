from typing import Literal, TypeAlias

PyLiteralType: TypeAlias = Literal['bool', 'int', 'float', 'str', 'datetime', 'time', 'timedelta']

WhereOperator: TypeAlias = Literal[
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
