import pytest

from dbframe.validator import NameValidator


class TestNameValidator:
    def test_dbname(self):
        assert NameValidator.dbname('my_database') == 'my_database'
        assert NameValidator.dbname('My_Database') == 'my_database'
        assert NameValidator.table('my_database_1') == 'my_database_1'
        invalid_dbnames = [
            'my database',
            '123my_database',
            'my_database?',
            'my_database!',
            'my_database@',
            'my_database#',
            'my_database$',
            'my_database%',
            'my_database^',
            'my_database&',
            'my_database*',
        ]
        for dbname in invalid_dbnames:
            pytest.raises(ValueError, NameValidator.dbname, dbname)

    def test_schema(self):
        assert NameValidator.schema('my_schema') == 'my_schema'
        assert NameValidator.schema('My_Schema') == 'my_schema'
        assert NameValidator.schema('my_schema_1') == 'my_schema_1'
        invalid_schemas = [
            'my schema',
            '123my_schema',
            'my_schema?',
            'my_schema!',
            'my_schema@',
            'my_schema#',
            'my_schema$',
            'my_schema%',
            'my_schema^',
            'my_schema&',
            'my_schema*',
        ]
        for schema in invalid_schemas:
            pytest.raises(ValueError, NameValidator.schema, schema)

    def test_table(self):
        assert NameValidator.table('my_table') == 'my_table'
        assert NameValidator.table('My_Table') == 'my_table'
        assert NameValidator.table('my_table_1') == 'my_table_1'
        invalid_tables = [
            'my table',
            '123my_table',
            'my_table?',
            'my_table!',
            'my_table@',
            'my_table#',
            'my_table$',
            'my_table%',
            'my_table^',
            'my_table&',
            'my_table*',
        ]
        for table in invalid_tables:
            pytest.raises(ValueError, NameValidator.table, table)

    def test_column(self):
        assert NameValidator.column('my_column') == 'my_column'
        assert NameValidator.column('My_column') == 'my_column'
        assert NameValidator.column('my_column_1') == 'my_column_1'
        assert NameValidator.column('_my_column') == '_my_column'
        assert NameValidator.column('my_column_') == 'my_column_'
        invalid_columns = [
            'my column',
            '123my_column',
            'my_column?',
            'my_column!',
            'my_column@',
            'my_column#',
            'my_column$',
            'my_column%',
            'my_column^',
            'my_column&',
            'my_column*',
        ]
        for column in invalid_columns:
            pytest.raises(ValueError, NameValidator.column, column)
