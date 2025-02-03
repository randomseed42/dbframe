import unittest
from io import StringIO
from unittest.mock import patch

from dbframe import SQLiteDFHandler


class TestSQLiteHandler(unittest.TestCase):
    def test_sqlite_handler_insert(self):
        db = SQLiteDFHandler()
        with patch('sys.stdout', new=StringIO()) as stdout:
            db.insert()
            self.assertEqual(stdout.getvalue().strip(), 'sqlite sqlite:///:memory: insert')


if __name__ == '__main__':
    unittest.main()
