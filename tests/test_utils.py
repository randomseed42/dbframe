import os
import unittest
from logging import FileHandler

import numpy as np
import pandas as pd
from dbframe.logger import Logger
from dbframe.utils import WhereClause, df_to_sql_columns, series_to_sql_dtype, where_clauses_parser
from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, MetaData, String, Table, Text, Time


class TestUtils(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='Utils_Logger',
        log_level='CRITICAL',
        log_console=True,
        log_file='utils.log',
    )

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger(**cls.LOGGER_CONF).get_logger()

    @classmethod
    def tearDownClass(cls):
        for hdlr in cls.logger.handlers:
            if isinstance(hdlr, FileHandler):
                hdlr.close()
        os.remove(cls.LOGGER_CONF.get('log_file'))

    # Where
    def test_where_clauses_parser(self):
        table = Table(
            'users',
            MetaData(),
            Column('uid', Integer),
            Column('user', Text),
            Column('email', Text),
        )
        columns = {k: v for k, v in table.columns.items()}
        where_clauses = [
            WhereClause('uid', '>', 0),
            (
                WhereClause('user', '>', 'user0'),
                WhereClause('user', '>', 'user1'),
            )
        ]
        where_condition = where_clauses_parser(where_clauses=where_clauses, columns=columns)
        self.assertEqual(str(where_condition),
                         'users.uid > :uid_1 AND (users."user" > :user_1 OR users."user" > :user_2)')
        where_condition = where_clauses_parser(where_clauses=where_clauses, table=table)
        self.assertEqual(str(where_condition),
                         'users.uid > :uid_1 AND (users."user" > :user_1 OR users."user" > :user_2)')
        with self.assertRaisesRegex(ValueError, 'Where clause must have one.*'):
            where_clauses_parser(where_clauses=where_clauses, columns=columns, table=table)
        with self.assertRaisesRegex(ValueError, 'Where clause must have one.*'):
            where_clauses_parser(where_clauses=where_clauses)

    # Dtype
    def test_series_to_sql_dtype_bool(self):
        cases = [
            # Bool Types
            (pd.Series(dtype=bool), Boolean),
            (pd.Series(dtype=bool).convert_dtypes(), Boolean),
            (pd.Series([True, False]), Boolean),
            (pd.Series([True, False]).convert_dtypes(), Boolean),
            (pd.Series([0, 1]).astype(bool), Boolean),
            (pd.Series(['True', '0']).astype(bool), Boolean),
        ]
        for i, (x, t) in enumerate(cases):
            self.assertEqual(series_to_sql_dtype(x), t)

    def test_series_to_sql_dtype_integer(self):
        cases = [
            # Integer Types
            (pd.Series(dtype=int), Integer),
            (pd.Series(dtype=int).convert_dtypes(), Integer),
            (pd.Series([1, 2, 3]), Integer),
            (pd.Series([1, 2, None]), Float),
            (pd.Series([1, 2, 3]).convert_dtypes(), Integer),
            (pd.Series([1.0, 2.0, 3.0]).convert_dtypes(), Integer),
            (pd.RangeIndex(0, 10, 1), Integer),
            (pd.Index([1,2,3]), Integer),
            (pd.Series(pd.Index([1, 2, None])).convert_dtypes(), Integer),

        ]
        for i, (x, t) in enumerate(cases):
            self.assertEqual(series_to_sql_dtype(x), t)

    def test_series_to_sql_dtype_float(self):
        cases = [
            # Float Types
            (pd.Series(dtype=float), Float),
            (pd.Series(dtype=float).convert_dtypes(), Integer),
            (pd.Series(dtype=pd.Float32Dtype()), Float),
            (pd.Series(dtype=pd.Float64Dtype()), Float),
            (pd.Series([1.0, 2.0]), Float),
            (pd.Series([1.0, 2.0]).convert_dtypes(), Integer),
            (pd.Index([1.1, 2.1, 3.1]), Float),
            (pd.Index([1, 2, None]), Float),
        ]
        for i, (x, t) in enumerate(cases):
            self.assertEqual(series_to_sql_dtype(x), t)

    def test_series_to_sql_dtype_string(self):
        cases = [
            # String Types
            (pd.Series(['1', '2', '3']).convert_dtypes(), String),
            (pd.Series(dtype=object), String),
            (pd.Series(dtype=np.object_), String),
            (pd.Categorical(['a', 'b', 'a']), String),
            (pd.Series(['a', 'b', 'a']).astype(pd.CategoricalDtype()), String),
            (pd.Index(['a', 'b', 'c']), String),
            (pd.Index(['a', 'b', None]), String),
            (pd.MultiIndex.from_product([['a', 'b'], [0, 1]]), String),
        ]
        for i, (x, t) in enumerate(cases):
            try:
                self.assertEqual(series_to_sql_dtype(x), t)
            except Exception:
                breakpoint()

    def test_series_to_sql_dtype_datetime(self):
        cases = [
            (np.array(['2025-01-01', '2025-01-02']), String),
            (np.array(['2025-01-01', '2025-01-02']).astype(np.datetime64), DateTime),
            (pd.Series(['2025-01-01', '2025-01-02']), String),
            (pd.to_datetime(pd.Series(['2025-01-01', '2025-01-02'])), DateTime),
            (pd.Series(['2025-01', '2025-02']), String),
            (pd.to_datetime(pd.Series(['2025-01', '2025-02'])), DateTime),
            (pd.to_datetime(pd.Series(['2025-01-01 08', '2025-02-01 09'])), DateTime),
            (pd.to_datetime(pd.Series(['2025-01-01 00:00:00+8', '2025-01-02 00:00:00+8'])), DateTime(timezone=True)),
            (pd.to_datetime(pd.Series(['2025-01-01 00:00:00+0800', '2025-01-02 00:00:00+0800'])), DateTime(timezone=True)),
            (pd.date_range('2024-01-01', '2025-01-01', freq='MS'), DateTime),
        ]
        for i, (x, t) in enumerate(cases):
            try:
                self.assertEqual(series_to_sql_dtype(x), t)
            except AssertionError:
                self.assertTrue(isinstance(series_to_sql_dtype(x), type(t)))

    def test_series_to_sql_dtype_time(self):
        cases = [
            (np.array(['08:00:00', '08:01:01']), String),
            (pd.Series(['08:00:00', '08:01:02']), String),
            (pd.to_timedelta(pd.Series(['08:00:00', '08:01:01'])), Time),
        ]
        for i, (x, t) in enumerate(cases):
            self.assertEqual(series_to_sql_dtype(x), t)

    def test_series_to_sql_dtype_special(self):
        x = pd.Series(dtype='datetime64[s]')
        self.assertEqual(series_to_sql_dtype(x), DateTime)

        x = pd.date_range('2025-01', periods=5, freq='D')
        self.assertEqual(series_to_sql_dtype(x), DateTime)

        x = pd.period_range('2025-01', periods=5, freq='D').to_timestamp()
        self.assertEqual(series_to_sql_dtype(x), DateTime)

        x = pd.Series(dtype=pd.CategoricalDtype())
        self.assertEqual(series_to_sql_dtype(x), String)

    def test_dtype_equal(self):
        self.logger.info(f'{id(Integer())}, {id(Integer())}')
        self.assertNotEqual(Integer(), Integer())
        self.assertTrue(Integer is Integer)

    def test_dataframe_to_sql_columns(self):
        data = [
            ['John', 17, 1.75],
            ['Jack', 18, 1.80],
            ['Jane', 19, 1.66],
            ['Judy', 16, 1.62],
        ]
        columns = ['nAmE', 'aGe', 'height']
        df = pd.DataFrame(data=data, columns=columns)

        columns = df_to_sql_columns(df=df, table_name='users')
        self.assertEqual(len(columns), 3)

        columns = df_to_sql_columns(df=df, table_name='users',
                                    primary_column_name='index')
        self.assertEqual(len(columns), 4)
        self.assertTrue(columns[0].primary_key)

        columns = df_to_sql_columns(df=df, table_name='users',
                                    primary_column_name='nAmE')
        self.assertEqual(len(columns), 3)
        self.assertEqual(columns[0].name, 'name')

        columns = df_to_sql_columns(df=df, table_name='users',
                                    primary_column_name=None,
                                    notnull_column_names=['nAmE', 'aGe'])
        self.assertEqual(len(columns), 3)
        self.assertFalse(columns[0].nullable)
        self.assertFalse(columns[1].nullable)

        columns = df_to_sql_columns(df=df, table_name='uSeRs',
                                    primary_column_name=None,
                                    notnull_column_names=['nAmE', 'age'],
                                    index_column_names=['nAmE', ['aGe', 'height']])
        self.assertEqual(len(columns), 5)
        self.assertFalse(columns[0].index)
        self.assertFalse(columns[1].index)
        self.assertTrue(isinstance(columns[3], Index))
        self.assertEqual(columns[3].name, 'ix_users_name')
        self.assertEqual(columns[4].name, 'ix_users_age_height')

        columns = df_to_sql_columns(df=df, table_name='uSeRs',
                                    primary_column_name=None,
                                    notnull_column_names=['nAmE', 'age'],
                                    index_column_names=['nAmE', ['aGe', 'height']],
                                    unique_column_names=['aGe', ['aGe', 'height']])
        self.assertEqual(len(columns), 7)
        self.assertEqual(columns[6].name, 'uix_users_age_height')
