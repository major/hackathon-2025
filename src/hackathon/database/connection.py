"""Database connection management."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from hackathon.config import get_settings
from hackathon.models.database import Base

settings = get_settings()

# Create engine
engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize the database by creating all tables and enabling pgvector extension.

    This function should be called once at application startup.
    """
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        SQLAlchemy Session instance

    Example:
        >>> for db in get_db():
        ...     # Use db session
        ...     pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope with automatic commit/rollback.

    This context manager automatically commits on success and rolls back on exceptions.

    Yields:
        SQLAlchemy Session instance

    Example:
        >>> from hackathon.database import session_scope
        >>> with session_scope() as db:
        ...     # Your database operations here
        ...     create_document(db, doc_data)
        ...     # Automatically commits on exit if no exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """
    Reset the database by dropping and recreating all tables.

    Warning:
        This will delete all data in the database!
    """
    Base.metadata.drop_all(bind=engine)
    init_db()
