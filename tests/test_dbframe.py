import unittest

from dbframe import DBHandler


class TestDBHandler(unittest.TestCase):
    def test_invalid_db_type(self):
        with self.assertRaises(ValueError) as context:
            DBHandler('mysql', 'dummy_conn_str')
        self.assertEqual(str(context.exception), 'Unsupported database type: mysql')
    
    def test_pg_handler_select(self):
        db = DBHandler(db_type='postgres', conn_str='pg_conn_str')
        db.select()
    
    def test_sqlite_handler_insert(self):
        db = DBHandler(db_type='sqlite', conn_str='sqlite_conn_str')
        db.insert()


if __name__ == '__main__':
    unittest.main()
