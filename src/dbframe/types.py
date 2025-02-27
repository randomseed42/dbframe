from typing import Literal, TypeAlias

LogLevelType: TypeAlias = Literal[
    'CRITICAL',
    'ERROR',
    'WARNING',
    'INFO',
    'DEBUG',
    'critical',
    'error',
    'warning',
    'info',
    'debug',
]

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
