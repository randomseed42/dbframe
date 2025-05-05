import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.exc import IntegrityError, OperationalError

from dbframe import Order, Pg, PgDF, Where


PG_CONN = dict(
    host='127.0.0.1',
    port=5432,
    user='postgres',
    password='postgres',
    dbname='postgres',
)


class TestPgDatabase:
    def test_get_url(self):
        pg = Pg(**PG_CONN)
        assert pg.get_url() == 'postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres'

    def test_validate_conn(self):
        pg = Pg(**PG_CONN)
        assert pg.validate_conn()
        conn_kw = PG_CONN.copy()
        conn_kw.update(dbname='non_existent_db')
        pg = Pg(**conn_kw)
        pytest.raises(OperationalError, pg.validate_conn)
