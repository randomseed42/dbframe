from typing import Literal, TypeAlias

Operator: TypeAlias = Literal[
    '=',
    '==',
    '!=',
    '<',
    '<=',
    '>',
    '>=',
    'in',
    'not in',
    'not_in',
    'like',
    'not like',
    'not_like',
    'ilike',
    'not ilike',
    'not_ilike',
    'between',
    'is',
    'is_',
    'is not',
    'is_not',
]


OPERATOR_MAPPER = {
    '=': '__eq__',
    '==': '__eq__',
    '!=': '__ne__',
    '<': '__lt__',
    '<=': '__le__',
    '>': '__gt__',
    '>=': '__ge__',
    'in': 'in_',
    'not in': 'not_in',
    'not_in': 'not_in',
    'like': 'like',
    'not like': 'not_like',
    'not_like': 'not_like',
    'ilike': 'ilike',
    'not ilike': 'not_ilike',
    'not_ilike': 'not_ilike',
    'between': 'between',
    'is': 'is_',
    'is_': 'is_',
    'is not': 'is_not',
    'is_not': 'is_not',
}


def get_oper(oper: Operator) -> str:
    return OPERATOR_MAPPER[oper]
