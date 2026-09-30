"""One place that turns DB_* settings into a connection URL.

Six call sites built this as an f-string —

    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

— which is this codebase's characteristic failure mode in one line. A password
containing '@', '/', ':', '#' or '?' does not raise: it re-splits the URL, so
the driver is handed a different host or database and the error names a machine
nobody typed. The same is true of a socket path in DB_HOST, whose slashes make
the authority section meaningless. And the string carries the password, so
printing the URL in an error message — app_v28.py did, in the one place that
runs when the connection fails — puts it on screen.

URL.create() takes each field as a value rather than as text to be re-parsed,
so none of that can happen, and str() on the result masks the password.

An empty password becomes None rather than "", because libpq consults
~/.pgpass and $PGPASSWORD only when no password is supplied, and on this
cluster that is how a Slurm job authenticates: home is shared, the server's own
socket connection needs no password at all.
"""

from __future__ import annotations

import os

from sqlalchemy.engine import URL

#: Matches tile_server_v2_.py, so a shell configured for the server needs no
#: extra setup anywhere else.
DEFAULT_USER = "vpandya"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5432"
DEFAULT_DATABASE = "hpl_kb"

DRIVER = "postgresql+psycopg2"


def database_url(database: str | None = None, *, user: str | None = None,
                 password: str | None = None, host: str | None = None,
                 port: str | int | None = None) -> URL:
    """The URL for one database. Anything not passed comes from the environment.

    Returns a URL rather than a string so callers cannot reintroduce the
    quoting problem by formatting it back together.
    """
    user = user if user is not None else os.getenv("DB_USER", DEFAULT_USER)
    password = password if password is not None else os.getenv("DB_PASS", "")
    host = host if host is not None else os.getenv("DB_HOST", DEFAULT_HOST)
    port = port if port is not None else os.getenv("DB_PORT", DEFAULT_PORT)
    database = database if database is not None else os.getenv("DB_NAME",
                                                               DEFAULT_DATABASE)

    # A non-numeric port is left off entirely rather than crashed on: libpq then
    # takes $PGPORT or its own default, which beats a ValueError at import time
    # in a module the UI imports at startup.
    try:
        port_number: int | None = int(str(port))
    except (TypeError, ValueError):
        port_number = None

    return URL.create(
        DRIVER,
        username=user or None,
        password=password or None,
        host=host or None,
        port=port_number,
        database=database,
    )


def safe_text(url: URL) -> str:
    """The URL as text with the password masked, for logs and the UI."""
    return url.render_as_string(hide_password=True)
