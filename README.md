# dbframe
A python package to simplify CRUD operations between SQL databases and pandas dataframe.

## Install

The preferred way to install dbframe is via pip

```commandline
pip install dbframe
```

## Basic Usage

### SQLite

```python
import pandas as pd
from dbframe import SQLiteDFHandler, WhereClause, OrderByClause

sqh = SQLiteDFHandler(db_path='my.db', log_level='info')
df = pd.DataFrame(data=[
    ['Jack', 17, 1.78],
    ['Jane', 16, 1.65],
    ['John', 18, 1.83],
])
sqh.df_create_table(df=df, table_name='students')

df_from_db = sqh.df_select_rows(
    table_name='students'
)
```

### Postgresql

```python
import pandas as pd
from dbframe import PGHandler, PGDFHandler, WhereClause, OrderByClause

sqh = SQLiteDFHandler(db_path='my.db', log_level='info')
df = pd.DataFrame(data=[
    ['Jack', 17, 1.78],
    ['Jane', 16, 1.65],
    ['John', 18, 1.83],
])
sqh.df_create_table(df=df, table_name='students')
```