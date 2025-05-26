import datetime
import decimal
import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import (
    ARRAY,
    BIGINT,
    BINARY,
    BLOB,
    BOOLEAN,
    CHAR,
    CLOB,
    DATE,
    DATETIME,
    DOUBLE,
    DOUBLE_PRECISION,
    FLOAT,
    INT,
    INTEGER,
    JSON,
    NCHAR,
    NVARCHAR,
    REAL,
    SMALLINT,
    TEXT,
    TIME,
    TIMESTAMP,
    UUID,
    VARBINARY,
    VARCHAR,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Double,
    Float,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    TupleType,
    Unicode,
    UnicodeText,
    Uuid,
)
from sqlalchemy.sql.sqltypes import (
    BOOLEANTYPE,
    DATETIME_TIMEZONE,
    TIME_TIMEZONE,
    NullType,
)

from dbframe.dtype import df_to_schema_items, dtype_sql_to_py, dtype_sql_to_str, object_to_sql_dtype


class TestDtype:
    def test_dtype_sql_to_py(self):
        assert dtype_sql_to_py(Boolean) is bool
        assert dtype_sql_to_py(BOOLEAN) is bool
        assert dtype_sql_to_py(BOOLEANTYPE) is bool

        assert dtype_sql_to_py(BIGINT) is int
        assert dtype_sql_to_py(BigInteger) is int
        assert dtype_sql_to_py(INT) is int
        assert dtype_sql_to_py(Integer) is int
        assert dtype_sql_to_py(INTEGER) is int
        assert dtype_sql_to_py(SMALLINT) is int
        assert dtype_sql_to_py(SmallInteger) is int

        assert dtype_sql_to_py(CHAR) is str
        assert dtype_sql_to_py(CLOB) is str
        assert dtype_sql_to_py(NCHAR) is str
        assert dtype_sql_to_py(NVARCHAR) is str
        assert dtype_sql_to_py(String) is str
        assert dtype_sql_to_py(Text) is str
        assert dtype_sql_to_py(TEXT) is str
        assert dtype_sql_to_py(Unicode) is str
        assert dtype_sql_to_py(UnicodeText) is str
        assert dtype_sql_to_py(VARCHAR) is str

        assert dtype_sql_to_py(DOUBLE_PRECISION) is float
        assert dtype_sql_to_py(Double) is float
        assert dtype_sql_to_py(DOUBLE) is float
        assert dtype_sql_to_py(Float) is float
        assert dtype_sql_to_py(FLOAT) is float
        assert dtype_sql_to_py(Numeric(asdecimal=False)) is float
        assert dtype_sql_to_py(Numeric) is decimal.Decimal
        assert dtype_sql_to_py(REAL) is float

        assert dtype_sql_to_py(Date) is datetime.date
        assert dtype_sql_to_py(DATE) is datetime.date
        assert dtype_sql_to_py(DATETIME_TIMEZONE) is datetime.datetime
        assert dtype_sql_to_py(DateTime) is datetime.datetime
        assert dtype_sql_to_py(DATETIME) is datetime.datetime
        assert dtype_sql_to_py(TIME_TIMEZONE) is datetime.time
        assert dtype_sql_to_py(Time) is datetime.time
        assert dtype_sql_to_py(TIME) is datetime.time
        assert dtype_sql_to_py(TIMESTAMP) is datetime.datetime

        assert dtype_sql_to_py(BINARY) is bytes
        assert dtype_sql_to_py(BLOB) is bytes
        assert dtype_sql_to_py(LargeBinary) is bytes
        assert dtype_sql_to_py(VARBINARY) is bytes

        assert dtype_sql_to_py(Uuid(as_uuid=False)) is str
        assert dtype_sql_to_py(Uuid) is uuid.UUID
        assert dtype_sql_to_py(UUID) is uuid.UUID

        assert dtype_sql_to_py(ARRAY(item_type=str)) is list
        assert dtype_sql_to_py(ARRAY(item_type=String)) is list
        assert dtype_sql_to_py(JSON) is dict

        pytest.raises(NotImplementedError, dtype_sql_to_py, NullType)
        pytest.raises(NotImplementedError, dtype_sql_to_py, TupleType)
        pytest.raises(TypeError, dtype_sql_to_py, ARRAY)

    def test_dtype_sql_to_str(self):
        assert dtype_sql_to_str(Boolean) == 'boolean'.lower()
        assert dtype_sql_to_str(BOOLEAN) == 'BOOLEAN'.lower()
        assert dtype_sql_to_str(BOOLEANTYPE) == 'boolean'.lower()

        assert dtype_sql_to_str(BIGINT) == 'BIGINT'.lower()
        assert dtype_sql_to_str(BigInteger) == 'big_integer'.lower()
        assert dtype_sql_to_str(INT) == 'INTEGER'.lower()
        assert dtype_sql_to_str(Integer) == 'integer'.lower()
        assert dtype_sql_to_str(INTEGER) == 'INTEGER'.lower()
        assert dtype_sql_to_str(SMALLINT) == 'SMALLINT'.lower()
        assert dtype_sql_to_str(SmallInteger) == 'small_integer'.lower()

        assert dtype_sql_to_str(CHAR) == 'CHAR'.lower()
        assert dtype_sql_to_str(CLOB) == 'CLOB'.lower()
        assert dtype_sql_to_str(NCHAR) == 'NCHAR'.lower()
        assert dtype_sql_to_str(NVARCHAR) == 'NVARCHAR'.lower()
        assert dtype_sql_to_str(String) == 'string'.lower()
        assert dtype_sql_to_str(Text) == 'text'.lower()
        assert dtype_sql_to_str(TEXT) == 'TEXT'.lower()
        assert dtype_sql_to_str(Unicode) == 'unicode'.lower()
        assert dtype_sql_to_str(UnicodeText) == 'unicode_text'.lower()
        assert dtype_sql_to_str(VARCHAR) == 'VARCHAR'.lower()

        assert dtype_sql_to_str(DOUBLE_PRECISION) == 'DOUBLE_PRECISION'.lower()
        assert dtype_sql_to_str(Double) == 'double'.lower()
        assert dtype_sql_to_str(DOUBLE) == 'DOUBLE'.lower()
        assert dtype_sql_to_str(Float) == 'float'.lower()
        assert dtype_sql_to_str(FLOAT) == 'FLOAT'.lower()
        assert dtype_sql_to_str(Numeric(asdecimal=False)) == 'numeric'.lower()
        assert dtype_sql_to_str(Numeric) == 'numeric'.lower()
        assert dtype_sql_to_str(REAL) == 'REAL'.lower()

        assert dtype_sql_to_str(Date) == 'date'.lower()
        assert dtype_sql_to_str(DATE) == 'DATE'.lower()
        assert dtype_sql_to_str(DATETIME_TIMEZONE) == 'datetime'.lower()
        assert dtype_sql_to_str(DateTime) == 'datetime'.lower()
        assert dtype_sql_to_str(DATETIME) == 'DATETIME'.lower()
        assert dtype_sql_to_str(TIME_TIMEZONE) == 'time'.lower()
        assert dtype_sql_to_str(Time) == 'time'.lower()
        assert dtype_sql_to_str(TIME) == 'TIME'.lower()
        assert dtype_sql_to_str(TIMESTAMP) == 'TIMESTAMP'.lower()

        assert dtype_sql_to_str(BINARY) == 'BINARY'.lower()
        assert dtype_sql_to_str(BLOB) == 'BLOB'.lower()
        assert dtype_sql_to_str(LargeBinary) == 'large_binary'.lower()
        assert dtype_sql_to_str(VARBINARY) == 'VARBINARY'.lower()

        assert dtype_sql_to_str(Uuid(as_uuid=False)) == 'uuid'.lower()
        assert dtype_sql_to_str(Uuid) == 'uuid'.lower()
        assert dtype_sql_to_str(UUID) == 'UUID'.lower()

        assert dtype_sql_to_str(ARRAY(item_type=str)) == 'ARRAY'.lower()
        assert dtype_sql_to_str(ARRAY(item_type=String)) == 'ARRAY'.lower()
        assert dtype_sql_to_str(JSON) == 'JSON'.lower()

        assert dtype_sql_to_str(NullType) == 'null'.lower()
        pytest.raises(AttributeError, dtype_sql_to_str, TupleType)

    def test_object_to_sql_dtype(self):
        assert object_to_sql_dtype(bool) is Boolean
        assert object_to_sql_dtype(int) is Integer
        assert object_to_sql_dtype(str) is String
        assert object_to_sql_dtype(float) is Float
        assert object_to_sql_dtype(datetime.datetime) is DateTime
        assert object_to_sql_dtype(datetime.date) is Date
        assert object_to_sql_dtype(datetime.time) is Time
        assert object_to_sql_dtype(bytes) is LargeBinary
        assert object_to_sql_dtype(uuid.UUID) is Uuid
        assert object_to_sql_dtype(list) is ARRAY
        assert object_to_sql_dtype(dict) is JSON

        assert object_to_sql_dtype('bool') is Boolean
        assert object_to_sql_dtype('boolean') is Boolean
        assert object_to_sql_dtype('int') is Integer
        assert object_to_sql_dtype('bigint') is Integer
        assert object_to_sql_dtype('big_integer') is Integer
        assert object_to_sql_dtype('integer') is Integer
        assert object_to_sql_dtype('smallint') is Integer
        assert object_to_sql_dtype('small_integer') is Integer
        assert object_to_sql_dtype('str') is String
        assert object_to_sql_dtype('char') is String
        assert object_to_sql_dtype('clob') is String
        assert object_to_sql_dtype('nchar') is String
        assert object_to_sql_dtype('nvarchar') is String
        assert object_to_sql_dtype('string') is String
        assert object_to_sql_dtype('text') is String
        assert object_to_sql_dtype('unicode') is String
        assert object_to_sql_dtype('unicode_text') is String
        assert object_to_sql_dtype('varchar') is String
        assert object_to_sql_dtype('float') is Float
        assert object_to_sql_dtype('double_precision') is Float
        assert object_to_sql_dtype('double') is Float
        assert object_to_sql_dtype('numeric') is Float
        assert object_to_sql_dtype('real') is Float
        assert object_to_sql_dtype('datetime') is DateTime
        assert object_to_sql_dtype('timestamp') is DateTime
        assert object_to_sql_dtype('date') is Date
        assert object_to_sql_dtype('time') is Time
        assert object_to_sql_dtype('binary') is LargeBinary
        assert object_to_sql_dtype('blob') is LargeBinary
        assert object_to_sql_dtype('large_binary') is LargeBinary
        assert object_to_sql_dtype('varbinary') is LargeBinary
        assert object_to_sql_dtype('uuid') is Uuid
        assert object_to_sql_dtype('array') is ARRAY
        assert object_to_sql_dtype('dict') is JSON

        assert object_to_sql_dtype(np.array([True, False, np.nan, None], dtype=np.bool)) is Boolean
        assert object_to_sql_dtype(np.array([True, False, np.nan])) is Float
        assert object_to_sql_dtype(np.array([True, False, np.nan, None])) is JSON
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32):
            assert object_to_sql_dtype(np.array([np.iinfo(dtype).min, np.iinfo(dtype).max], dtype=dtype)) is Integer
            assert object_to_sql_dtype(np.array([np.iinfo(dtype).min, np.iinfo(dtype).max])) is Integer
        assert object_to_sql_dtype(np.array([np.iinfo(np.uint64).min, 2**63 - 1], dtype=np.uint64)) is Integer
        assert object_to_sql_dtype(np.array([np.iinfo(np.uint64).min, 2**63 - 1])) is Integer
        # assert object_to_sql_dtype(np.array([np.iinfo(np.uint64).min, np.iinfo(np.uint64).max], dtype=np.uint64)) is Float  # possible numpy bug
        assert object_to_sql_dtype(np.array([np.iinfo(np.uint64).min, np.iinfo(np.uint64).max])) is Float  # possible numpy bug
        assert object_to_sql_dtype(np.array(['a', '中文', '🚀'], dtype=np.str_)) is String
        assert object_to_sql_dtype(np.array(['a', '中文', '🚀'])) is String
        for dtype in (np.float16, np.float32, np.float64):
            assert object_to_sql_dtype(np.array([np.finfo(dtype).min, np.finfo(dtype).max, np.nan, None], dtype=dtype)) is Float
            assert object_to_sql_dtype(np.array([np.finfo(dtype).min, np.finfo(dtype).max, np.nan, None])) is JSON
        assert object_to_sql_dtype(np.array(['2025-01-01'], dtype='datetime64[D]')) is DateTime
        assert object_to_sql_dtype(np.array([np.datetime64('2025-01-01')], dtype='datetime64[D]')) is DateTime
        assert object_to_sql_dtype(np.datetime64('2025-01-01')) is DateTime
        assert object_to_sql_dtype(np.array([b'xyz'], dtype=np.bytes_)) is LargeBinary
        assert object_to_sql_dtype(np.array([b'xyz'])) is LargeBinary
        assert object_to_sql_dtype(np.array([{'a': 1}], dtype=np.object_)) is JSON

        assert object_to_sql_dtype(pd.Series([True, False])) is Boolean
        assert object_to_sql_dtype(pd.Series([True, False, np.nan, None], dtype=pd.BooleanDtype())) is Boolean
        assert object_to_sql_dtype(pd.Series([True, False, np.nan, None]).convert_dtypes()) is Boolean
        assert object_to_sql_dtype(pd.Series([True, False, np.nan, None]).astype(bool)) is Boolean
        assert object_to_sql_dtype(pd.Series([True, False, np.nan, None]).astype('boolean')) is Boolean
        assert object_to_sql_dtype(pd.Series([True, False, np.nan, None])) is Boolean

        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2])) is Integer
        for dtype in (pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()):
            assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None], dtype=dtype)) is Integer
        for dtype in (pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype()):
            assert object_to_sql_dtype(pd.Series([0, 1, 2, np.nan, pd.NA, None], dtype=dtype)) is Integer
        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None], dtype=pd.Int64Dtype())) is Integer
        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None]).convert_dtypes()) is Integer
        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None]).astype('Int64')) is Integer
        pytest.raises(ValueError, pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None]).astype, 'int64')
        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, None])) is Float
        assert object_to_sql_dtype(pd.Series([-1, 0, 1, 2, np.nan, pd.NA, None])) is Integer

        assert object_to_sql_dtype(pd.Series(['a', '中文', '🚀'])) is String
        assert object_to_sql_dtype(pd.Series(['a', '中文', '🚀'])) is String
        assert object_to_sql_dtype(pd.Series(['a', '中文', {'emoji': '🚀'}])) is JSON

        assert object_to_sql_dtype(pd.Series([-1.5, 0, 0.5, np.nan, pd.NA, None])) is Float
        for dtype in (pd.Float32Dtype(), pd.Float64Dtype()):
            assert object_to_sql_dtype(pd.Series([-1.5, 0, 0.5, np.nan, pd.NA, None], dtype=dtype)) is Float

        assert object_to_sql_dtype(pd.Series(['2025-01-01 00:00:00'])) is String
        assert object_to_sql_dtype(pd.Series(['2025-01-01 00:00:00'], dtype='datetime64[ns]')) is DateTime
        assert object_to_sql_dtype(pd.Series(['2025-01-01 00:00:00']).astype('datetime64[ns]')) is DateTime
        assert object_to_sql_dtype(pd.to_datetime(pd.Series(['2025-01-01 00:00:00']))) is DateTime
        assert object_to_sql_dtype(pd.to_datetime(['2025-01-01 00:00:00'])) is DateTime
        assert object_to_sql_dtype(pd.Series(['2025-01'])) is String
        assert object_to_sql_dtype(pd.Series(['2025-01'], dtype='datetime64[ns]')) is DateTime
        assert object_to_sql_dtype(pd.Series(['2025-01'], dtype=pd.api.types.DatetimeTZDtype(tz='utc'))) == DATETIME_TIMEZONE
        assert object_to_sql_dtype(pd.to_datetime(pd.Series(['2025-01']))) is DateTime
        assert object_to_sql_dtype(pd.to_datetime(pd.Series(['2025-01-01T00:00:00']), format='%Y-%m-%dT%H:%M:%S')) is DateTime
        assert object_to_sql_dtype(pd.to_datetime(['2025-01-01 00:00:00']).tz_localize('UTC')) == DATETIME_TIMEZONE
        assert object_to_sql_dtype(pd.date_range('2025-01-01', '2025-01-03')) is DateTime
        assert object_to_sql_dtype(pd.date_range('2025-01-01', '2025-01-03').tz_localize('UTC')) == DATETIME_TIMEZONE
        assert object_to_sql_dtype(pd.Series(['00:00:00'])) is String
        pytest.raises(TypeError, object_to_sql_dtype, pd.Series(['00:00:00'], dtype='timedelta64[ns]'))

        assert object_to_sql_dtype(pd.Series([b'xyz'])) is JSON
        assert object_to_sql_dtype(pd.Series([b'xyz'], dtype=np.bytes_)) is LargeBinary
        assert object_to_sql_dtype(pd.Series([b'xyz'], dtype=bytes)) is LargeBinary

        df = pd.DataFrame(
            {'a': [1, 2, 3], 'b': [True, False, True], 'c': ['a', 'b', 'c'], 'd': ['2025-01-01', '2025-01-02', '2025-01-03']}
        )
        df['c'] = df['c'].astype(pd.CategoricalDtype())
        df['d'] = pd.to_datetime(df['d'])
        assert object_to_sql_dtype(df.index) is Integer
        assert object_to_sql_dtype(df['d']) is DateTime
        df = df.set_index('d')
        assert object_to_sql_dtype(df.index) is DateTime
        assert object_to_sql_dtype(df['c']) is String
        df = df.set_index('c')
        assert object_to_sql_dtype(df.index) is String


class TestDFtoSchema:
    def test_df_to_schema_items(self):
        df = pd.DataFrame(
            {'a': [1, 2, 3], 'b': [True, False, True], 'c': ['a', 'b', 'c'], 'd': ['2025-01-01', '2025-01-02', '2025-01-03']}
        )
        df['c'] = df['c'].astype(pd.CategoricalDtype())
        df['d'] = pd.to_datetime(df['d'])
        schema_items = df_to_schema_items(df, tb_nm='test_table')
        assert len(schema_items) == 4
        schema_items = df_to_schema_items(df, tb_nm='test_table', primary_col_nm='index')
        assert len(schema_items) == 5
        schema_items = df_to_schema_items(df, tb_nm='test_table', primary_col_nm='a', primary_col_autoinc=True)
        assert len(schema_items) == 4
        assert schema_items[0].name == 'a'
        assert isinstance(schema_items[0].type, Integer)
        assert schema_items[0].primary_key is True
        assert schema_items[0].autoincrement is True
        schema_items = df_to_schema_items(
            df,
            tb_nm='test_table',
            primary_col_nm='a',
            primary_col_autoinc=True,
            notnull_col_nms=['b'],
            index_col_nms=['d', ['b', 'c']],
            unique_col_nms=['b', ['d', 'c']],
        )
        assert len(schema_items) == 8
