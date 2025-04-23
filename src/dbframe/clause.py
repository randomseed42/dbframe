from typing import NamedTuple, Sequence

from sqlalchemy import Column, Table, and_, or_, true

from .operator import Operator, get_oper
from .validator import NameValidator


class Where(NamedTuple):
    column_name: str
    operator: Operator
    value: str | int | float | bool | Sequence | None


def where_parser(
    where: list[Where] | tuple[Where] | Where | None,
    columns: dict[str, Column] = None,
    table: Table = None,
):
    if where is None:
        return true()

    if columns is None or len(columns) == 0:
        if table is None:
            raise ValueError('Either columns or table must be provided.')
        columns = table.columns

    def _parse(_w: Where | None):
        if _w is None:
            return true()
        _column_name = NameValidator.column(_w.column_name)
        if _column_name not in columns:
            raise ValueError(f'Column {_column_name} not found in columns.')
        _column = columns[_column_name]

        if _w.operator == 'between':
            if _w.value is None or not isinstance(_w.value, Sequence) or len(_w.value) != 2:
                raise ValueError('Value for operator "between" must be a sequence of length 2.')
            _cond = getattr(_column, get_oper(_w.operator))(_w.value[0], _w.value[1])
        else:
            _cond = getattr(_column, get_oper(_w.operator))(_w.value)

        return _cond

    if isinstance(where, Where):
        return _parse(where)
    if isinstance(where, list):
        return and_(where_parser(where=_where, columns=columns, table=table) for _where in where)
    if isinstance(where, tuple):
        return or_(where_parser(where=_where, columns=columns, table=table) for _where in where)


class Order(NamedTuple):
    column_name: str
    ascending: bool = True


def order_parser(
    order: list[Order] | tuple[Order] | Order | None,
    columns: dict[str, Column] = None,
    table: Table = None,
):
    if order is None:
        return [None]

    if columns is None or len(columns) == 0:
        if table is None:
            raise ValueError('Either columns or table must be provided.')
        columns = table.columns

    if isinstance(order, Order):
        column_name = NameValidator.column(order.column_name)
        if order.ascending:
            return [columns.get(column_name).asc()]
        else:
            return [columns.get(column_name).desc()]

    if isinstance(order, list | tuple):
        orders = []
        for _o in order:
            orders.extend(order_parser(order=_o, columns=columns, table=table))
        return orders
