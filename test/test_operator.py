import pytest

from dbframe.operator import get_oper


class TestOperator:
    def test_get_oper(self):
        assert get_oper('==') == '__eq__'
        pytest.raises(KeyError, get_oper, 'invalid_operator')
