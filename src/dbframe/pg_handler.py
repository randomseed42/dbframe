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
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        self.host = host or os.getenv('PG_HOST', '127.0.0.1')
        self.port = str(port or os.getenv('PG_PORT', '5432'))
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = quote_plus(password or os.getenv('PG_PASS', 'postgres'))
        self.dbname = dbname or os.getenv('PG_DBNAME', 'postgres')
        self.logger = Logger(
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        ).get_logger()
        self.logger.critical(f'logger initialized {self.logger.name=} {id(self.logger)=}')

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

    def get_databases(self):
        stmt = text('SELECT datname FROM pg_catalog.pg_database;')
        engine = create_engine(self.url_default)
        with engine.connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchall()
            databases = [row[0] for row in rows]
            return databases


class PGDFHandler(PGHandler):
    def __init__(
            self,
            host: str = None,
            port: int | str = None,
            user: str = None,
            password: str = None,
            dbname: str = None,
            log_name: str = 'Logger',
            log_level: str = None,
            log_console: bool = False,
            log_file: str = None,
    ):
        super().__init__(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            log_name=log_name,
            log_level=log_level,
            log_console=log_console,
            log_file=log_file,
        )
