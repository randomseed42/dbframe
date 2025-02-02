import unittest
from io import StringIO
from unittest.mock import patch

import dbframe
from dbframe import PGDFHandler, SQLiteDFHandler

if 'src' in dbframe.__file__:
    print('Run development tests')
elif 'site-packages' in dbframe.__file__:
    print('Run post install tests')
else:
    raise ImportError(f'Please check package source: {dbframe.__file__}')


class TestDBHandler(unittest.TestCase):
    def test_pg_handler_select(self):
        db = PGDFHandler()
        with patch('sys.stdout', new=StringIO()) as stdout:
            db.select()
            self.assertEqual(
                stdout.getvalue().strip(),
                'postgres postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres select'
            )

    def test_sqlite_handler_insert(self):
        db = SQLiteDFHandler()
        with patch('sys.stdout', new=StringIO()) as stdout:
            db.insert()
            self.assertEqual(stdout.getvalue().strip(), 'sqlite sqlite:///:memory: insert')


if __name__ == '__main__':
    unittest.main()
