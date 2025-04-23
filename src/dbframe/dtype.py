import numpy as np
import pandas as pd
from typing import Literal, TypeAlias, Any
import datetime
from uuid import UUID as _python_UUID

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
from sqlalchemy.sql.sqltypes import TypeEngine


def dtype_sql_to_py(sql_dtype: TypeEngine | type) -> type:
    if isinstance(sql_dtype, type):
        return sql_dtype().python_type
    if isinstance(sql_dtype, TypeEngine):
        return sql_dtype.python_type


def dtype_sql_to_str(sql_dtype: TypeEngine | type) -> str:
    return sql_dtype.__visit_name__.lower()


DTYPE_PY_TO_SQL_MAPPER = {
    bool: Boolean,
    int: Integer,
    str: String,
    float: Float,
    datetime.datetime: DateTime,
    datetime.date: Date,
    datetime.time: Time,
    bytes: LargeBinary,
    _python_UUID: Uuid,
    list: ARRAY,
    dict: JSON,
}

DTYPE_STR_TO_SQL_MAPPER = {
    'bool': Boolean,
    'boolean': Boolean,
    'int': Integer,
    'bigint': Integer,
    'big_integer': Integer,
    'integer': Integer,
    'smallint': Integer,
    'small_integer': Integer,
    'str': String,
    'char': String,
    'clob': String,
    'nchar': String,
    'nvarchar': String,
    'string': String,
    'text': String,
    'unicode': String,
    'unicode_text': String,
    'varchar': String,
    'float': Float,
    'double_precision': Float,
    'double': Float,
    'numeric': Float,
    'real': Float,
    'datetime': DateTime,
    'timestamp': DateTime,
    'date': Date,
    'time': Time,
    'binary': LargeBinary,
    'blob': LargeBinary,
    'large_binary': LargeBinary,
    'varbinary': LargeBinary,
    'uuid': Uuid,
    'array': ARRAY,
    'dict': JSON,
}

DTYPE_NUMPY_TO_SQL_MAPPER = {
    np.bool: Boolean,
    np.int8: Integer,
    np.int16: Integer,
    np.int32: Integer,
    np.int64: Integer,
    np.uint8: Integer,
    np.uint16: Integer,
    np.uint32: Integer,
    np.uint64: Integer,
    np.str_: String,
    np.float16: Float,
    np.float32: Float,
    np.float64: Float,
    np.datetime64: DateTime,
    np.bytes_: LargeBinary,
    np.object_: JSON,
}

DTYPE_PANDAS_TO_SQL_MAPPER = {
    pd.BooleanDtype(): Boolean,
    pd.Int8Dtype(): Integer,
    pd.Int16Dtype(): Integer,
    pd.Int32Dtype(): Integer,
    pd.Int64Dtype(): Integer,
    pd.UInt8Dtype(): Integer,
    pd.UInt16Dtype(): Integer,
    pd.UInt32Dtype(): Integer,
    pd.UInt64Dtype(): Integer,
    pd.StringDtype(): String,
    pd.Float32Dtype(): Float,
    pd.Float64Dtype(): Float,
    pd.DatetimeTZDtype: DateTime,
    pd.DatetimeIndex: DateTime,
}


def object_to_sql_dtype(obj: Any) -> TypeEngine:
    if isinstance(obj, type):
        return DTYPE_PY_TO_SQL_MAPPER[obj]
    if isinstance(obj, str):
        return DTYPE_STR_TO_SQL_MAPPER[obj]
    if isinstance(obj, np.ndarray):
        return DTYPE_NUMPY_TO_SQL_MAPPER[obj.dtype.type]
    if np.isscalar(obj):
        return DTYPE_NUMPY_TO_SQL_MAPPER[type(obj)]
    if isinstance(obj, pd.Series):
        pd_type = obj.dtype
        if pd_type in DTYPE_PANDAS_TO_SQL_MAPPER:
            return DTYPE_PANDAS_TO_SQL_MAPPER[pd_type]
        np_type = pd_type.type
        if np_type is np.object_:
            inferred_pd_type = obj.convert_dtypes().dtype
            if inferred_pd_type is np.dtype('O'):
                return JSON
            if inferred_pd_type in DTYPE_PANDAS_TO_SQL_MAPPER:
                return DTYPE_PANDAS_TO_SQL_MAPPER[inferred_pd_type]
        if np_type in DTYPE_NUMPY_TO_SQL_MAPPER:
            return DTYPE_NUMPY_TO_SQL_MAPPER[np_type]
        raise TypeError(f'pandas series dtype {pd_type} not supported yet')
