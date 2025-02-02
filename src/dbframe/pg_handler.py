import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text

from .logger import Logger

logger = Logger('PG_Logger').get_logger()


class PGHandler:
    def __init__(
            self,
            host: str = None,
            port: int | str = None,
            user: str = None,
            password: str = None,
            dbname: str = None,
    ):
        self.host = host or os.getenv('PG_HOST', '127.0.0.1')
        self.port = str(port or os.getenv('PG_PORT', '5432'))
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = quote_plus(password or os.getenv('PG_PASS', 'postgres'))
        self.dbname = dbname or os.getenv('PG_DBNAME', 'postgres')
        self.url = self._generate_url(dbname=self.dbname)
        self.url_default = self._generate_url(dbname='postgres')
        self.engine = create_engine(
            self.url,
            executemany_mode='values_plus_batch',
            isolation_level='AUTOCOMMIT',
        )
        self._validate_connection()

    def _generate_url(self, dbname: str):
        url_template = 'postgresql+psycopg2://{}:{}@{}:{}/{}'
        return url_template.format(self.user, self.password, self.host, self.port, dbname)

    def _validate_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
                logger.info('Connection successful')
        except Exception as err:
            logger.error(str(err))
            raise ValueError(f'Connection failed: {self.host=} {self.port=} {self.user=} {self.dbname=}')

    def select(self):
        print(f'postgres {self.url} select')

    def insert(self):
        print(f'postgres {self.url} insert')


class PGDFHandler(PGHandler):
    def __init__(
            self,
            host: str = None,
            port: int | str = None,
            user: str = None,
            password: str = None,
            dbname: str = None,
    ):
        super().__init__(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
        )
