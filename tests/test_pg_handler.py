import unittest
from io import StringIO
from unittest.mock import patch

from dbframe import PGDFHandler


class TestPGHandler(unittest.TestCase):
    def test_pg_handler_select(self):
        db = PGDFHandler()
        with patch('sys.stdout', new=StringIO()) as stdout:
            db.select()
            self.assertEqual(
                stdout.getvalue().strip(),
                'postgres postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres select'
            )


if __name__ == '__main__':
    unittest.main()
