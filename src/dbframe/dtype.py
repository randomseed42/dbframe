import datetime
from base64 import b64decode
from typing import Any, Literal, Sequence
from uuid import UUID as _python_UUID

import numpy as np
import orjsonic
import pandas as pd
from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    Column,
    Constraint,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Time,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.sql.sqltypes import DATETIME_TIMEZONE, TypeEngine

from .validator import NameValidator


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
    pd.Timestamp: DateTime,
    pd.DatetimeTZDtype: DateTime,
    pd.DatetimeIndex: DateTime,
    pd.RangeIndex: Integer,
    pd.CategoricalDtype: String,
}


def _object_to_sql_dtype(obj: Any) -> TypeEngine:
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
        if isinstance(pd_type, pd.CategoricalDtype):
            return String
        sub_type = pd_type.type
        if sub_type is np.object_:
            inferred_pd_type = obj.convert_dtypes().dtype
            if inferred_pd_type is np.dtype('O'):
                if pd.isna(obj).all():
                    return String
                else:
                    return JSON
            if inferred_pd_type in DTYPE_PANDAS_TO_SQL_MAPPER:
                return DTYPE_PANDAS_TO_SQL_MAPPER[inferred_pd_type]
        if sub_type in DTYPE_NUMPY_TO_SQL_MAPPER:
            return DTYPE_NUMPY_TO_SQL_MAPPER[sub_type]
        if sub_type is pd.Timestamp:
            if pd_type.tz is None:
                return DateTime
            else:
                return DATETIME_TIMEZONE
        raise TypeError(f'pandas series dtype {pd_type} not supported yet.')
    if isinstance(obj, pd.Index):
        if isinstance(obj, pd.DatetimeIndex):
            if obj.tz is None:
                return DateTime
            else:
                return DATETIME_TIMEZONE
        if isinstance(obj, pd.PeriodIndex):
            return DateTime
        if isinstance(obj, pd.RangeIndex):
            return Integer
        if isinstance(obj, pd.CategoricalIndex):
            return String


def object_to_sql_dtype(obj: Any, dialect: Literal['sqlite', 'postgresql'] = None) -> TypeEngine:
    sql_dtype = _object_to_sql_dtype(obj)
    if sql_dtype is None:
        raise TypeError(f'unsupported dtype {type(obj)}.')
    if dialect == 'sqlite':
        if sql_dtype in (DateTime, Date, Time, DATETIME_TIMEZONE):
            return String
        if sql_dtype is JSON:
            return String
    return sql_dtype


def _df_convert_dtypes(df: pd.DataFrame, dialect: Literal['sqlite', 'postgresql'] = None) -> pd.DataFrame:
    byte_col_nms = []
    for col_nm in df.columns:
        try:
            df[col_nm] = df[col_nm].convert_dtypes()
            df[col_nm] = df[col_nm].where(pd.notnull(df[col_nm]), None)
            df[col_nm] = df[col_nm].replace('', None)
            if dialect != 'sqlite':
                continue
            if df[col_nm].dtype.type is not np.object_:
                continue
            if df[col_nm].dropna().empty:
                continue
            if isinstance(df[col_nm].dropna().sample(1).iloc[0], (dict, list, tuple, set)):
                df[col_nm] = df[col_nm].apply(orjsonic.dumps, return_str=True)
        except UnicodeDecodeError:
            byte_col_nms.append(col_nm)
    return byte_col_nms, df


def _df_to_primary_col(
    df: pd.DataFrame,
    primary_col_nm: str = None,
    primary_col_autoinc: Literal['auto', True, False] = 'auto',
    dialect: Literal['sqlite', 'postgresql'] = None,
) -> Column | None:
    if primary_col_nm is None:
        return
    if primary_col_nm == 'index':
        _primary_col_nm = 'uid'
        primary_col = df.index
    elif primary_col_nm not in df.columns:
        raise ValueError(f'primary column {primary_col_nm} not in dataframe.')
    else:
        _primary_col_nm = NameValidator.column(primary_col_nm)
        primary_col = df[primary_col_nm]
    if not primary_col.is_unique:
        raise ValueError(f'primary column {primary_col_nm} must be unique.')

    sql_dtype = object_to_sql_dtype(primary_col, dialect=dialect)

    if primary_col_autoinc == 'auto':
        return Column(_primary_col_nm, sql_dtype, primary_key=True, autoincrement='auto')
    elif primary_col_autoinc is True:
        if sql_dtype is not Integer:
            raise ValueError(f'primary column {primary_col_nm} must be integer for auto-increment.')
        return Column(_primary_col_nm, sql_dtype, primary_key=True, autoincrement=True)
    elif primary_col_autoinc is False:
        return Column(_primary_col_nm, sql_dtype, primary_key=True)
    else:
        raise ValueError(f'invalid primary_col_autoinc value {primary_col_autoinc}.')


def _df_to_index_constraint(df: pd.DataFrame, tb_nm: str, index_col_grp: str | Sequence[str]) -> Constraint:
    if isinstance(index_col_grp, str):
        index_col_grp = [index_col_grp]
    index_col_nms = []
    for index_col_nm in index_col_grp:
        if index_col_nm not in df.columns:
            raise ValueError(f'index column {index_col_nm} not in dataframe.')
        _index_col_nm = NameValidator.column(index_col_nm)
        if _index_col_nm in index_col_nms:
            raise ValueError(f'index column {index_col_nm} already in index group.')
        index_col_nms.append(_index_col_nm)
    index_nm = 'ix_{}_{}'.format(tb_nm, '_'.join(index_col_nms))
    index = Index(index_nm, *index_col_nms)
    return index


def _df_to_unique_constraint(df: pd.DataFrame, tb_nm: str, unique_col_grp: str | Sequence[str]) -> UniqueConstraint:
    if isinstance(unique_col_grp, str):
        unique_col_grp = [unique_col_grp]
    unique_col_nms = []
    for unique_col_nm in unique_col_grp:
        if unique_col_nm not in df.columns:
            raise ValueError(f'unique column {unique_col_nm} not in dataframe.')
        _unique_col_nm = NameValidator.column(unique_col_nm)
        if _unique_col_nm in unique_col_nms:
            raise ValueError(f'unique column {unique_col_nm} already in unique group.')
        unique_col_nms.append(_unique_col_nm)
    unique_nm = 'uix_{}_{}'.format(tb_nm, '_'.join(unique_col_nms))
    unique = UniqueConstraint(*unique_col_nms, name=unique_nm)
    return unique


def df_to_schema_items(
    df: pd.DataFrame,
    tb_nm: str,
    primary_col_nm: str = None,
    primary_col_autoinc: Literal['auto', True, False] = 'auto',
    notnull_col_nms: Sequence[str] = None,
    index_col_nms: Sequence[str | Sequence[str]] = None,
    unique_col_nms: Sequence[str | Sequence[str]] = None,
    dialect: Literal['sqlite', 'postgresql'] = None,
) -> Sequence[Column | Constraint | Index | UniqueConstraint]:
    _, df = _df_convert_dtypes(df, dialect=dialect)
    tb_nm = NameValidator.table(tb_nm)
    schema_items = []

    sql_col = _df_to_primary_col(df=df, primary_col_nm=primary_col_nm, primary_col_autoinc=primary_col_autoinc, dialect=dialect)
    if sql_col is not None:
        schema_items.append(sql_col)

    if notnull_col_nms is None:
        notnull_col_nms = []
    if index_col_nms is None:
        index_col_nms = []
    if unique_col_nms is None:
        unique_col_nms = []

    for col_nm in df.columns:
        if col_nm == primary_col_nm:
            continue
        _col_nm = NameValidator.column(col_nm)
        nullable = col_nm not in notnull_col_nms and _col_nm not in notnull_col_nms
        sql_dtype = object_to_sql_dtype(df[col_nm], dialect=dialect)
        sql_col = Column(_col_nm, sql_dtype, nullable=nullable)
        schema_items.append(sql_col)
    for index_col_nms_grp in index_col_nms:
        index = _df_to_index_constraint(df=df, tb_nm=tb_nm, index_col_grp=index_col_nms_grp)
        schema_items.append(index)
    for unique_col_nms_grp in unique_col_nms:
        unique = _df_to_unique_constraint(df=df, tb_nm=tb_nm, unique_col_grp=unique_col_nms_grp)
        schema_items.append(unique)
    return schema_items


def df_to_rows(df: pd.DataFrame, dialect: Literal['sqlite', 'postgresql'] = None) -> Sequence[dict]:
    df.columns = map(NameValidator.column, df.columns)
    byte_col_nms, df = _df_convert_dtypes(df, dialect=dialect)

    def decode_bytes(row: dict) -> dict:
        for col_nm in byte_col_nms:
            row[col_nm] = b64decode(row[col_nm])
        return row

    rows = orjsonic.loads(orjsonic.dumps(df.to_dict(orient='records')))
    if len(byte_col_nms) == 0:
        return rows
    rows = list(map(decode_bytes, rows))
    return rows
