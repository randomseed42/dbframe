import pytest
from sqlalchemy import Column, DateTime, Integer, true

from dbframe.clause import Order, Where, order_parser, where_parser


class TestClause:
    def test_where_parser(self):
        assert where_parser(where=None) is true()
        pytest.raises(ValueError, where_parser, where=Where('col', '==', 1))

        columns = {
            'col1': Column('col1', type_=Integer),
            'col2': Column('col2', type_=Integer),
            'col3': Column('col3', type_=DateTime),
        }
        where = Where('col1', '==', 2)
        cond = where_parser(where=where, cols=columns)
        assert str(cond) == 'col1 = :col1_1' and cond.right.value == 2

        where = [
            Where('col1', '>=', 2),
            (
                Where('col2', '<', 3),
                Where('col3', 'between', ('2025-01-01', '2025-02-01')),
            ),
        ]
        cond = where_parser(where=where, cols=columns)
        assert (
            str(cond.expression) == 'col1 >= :col1_1 AND (col2 < :col2_1 OR col3 BETWEEN :col3_1 AND :col3_2)'
            and cond.clauses[0].right.value == 2
            and cond.clauses[1].clauses[0].right.value == 3
            and cond.clauses[1].clauses[1].right.clauses[0].value == '2025-01-01'
            and cond.clauses[1].clauses[1].right.clauses[1].value == '2025-02-01'
        )

    def test_order_parser(self):
        assert order_parser(order=None) == [None]
        pytest.raises(ValueError, order_parser, order=Order('col'))

        columns = {
            'col1': Column('col1', type_=Integer),
            'col2': Column('col2', type_=Integer),
            'col3': Column('col3', type_=DateTime),
        }
        order = Order('col1')
        orders = order_parser(order=order, cols=columns)
        assert len(orders) == 1 and str(orders[0]) == 'col1 ASC'

        order = [Order('col1'), Order('col2', ascending=False)]
        orders = order_parser(order=order, cols=columns)
        assert len(orders) == 2 and str(orders[0]) == 'col1 ASC' and str(orders[1]) == 'col2 DESC'
