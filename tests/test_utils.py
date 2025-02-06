import os
import unittest
from logging import FileHandler

from dbframe.logger import Logger
from dbframe.utils import WhereClause, where_clauses_parser
from sqlalchemy import Table, TEXT, INTEGER, MetaData, Column


class TestUtils(unittest.TestCase):
    LOGGER_CONF = dict(
        log_name='Utils_Logger',
        log_level='DEBUG',
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

    def test_where_clauses_parser(self):
        table = Table(
            'users',
            MetaData(),
            Column('uid', INTEGER()),
            Column('user', TEXT()),
            Column('email', TEXT()),
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
