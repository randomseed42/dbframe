import re


class NameValidator:
    PATTERNS = {
        'dbname': r'^[a-zA-Z][a-zA-Z0-9_]*$',
        'schema': r'^[a-zA-Z][a-zA-Z0-9_]*$',
        'table': r'^[a-zA-Z][a-zA-Z0-9_]*$',
        'column': r'^[a-zA-Z_][a-zA-Z0-9_]*$',
        'index': r'^[a-zA-Z][a-zA-Z0-9_]*$',
    }

    @classmethod
    def _validate(cls, element: str, name: str) -> str:
        if not re.match(cls.PATTERNS[element], name):
            raise ValueError(f'The {element} {name} is invalid.')
        return name.lower()

    @classmethod
    def dbname(cls, name: str) -> str:
        return cls._validate(element='dbname', name=name)

    @classmethod
    def schema(cls, name: str) -> str:
        return cls._validate(element='schema', name=name)

    @classmethod
    def table(cls, name: str) -> str:
        return cls._validate(element='table', name=name)

    @classmethod
    def column(cls, name: str) -> str:
        return cls._validate(element='column', name=name)

    @classmethod
    def index(cls, name: str) -> str:
        return cls._validate(element='index', name=name)
