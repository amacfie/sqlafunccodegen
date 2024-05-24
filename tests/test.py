import unittest
from contextlib import asynccontextmanager

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine


from . import out_python, out_sqlalchemy


engine = create_async_engine(
    "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres",
    echo=True,
    poolclass=NullPool,
)


@asynccontextmanager
async def get_db_sesh():
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        yield session


class TestPython(unittest.IsolatedAsyncioTestCase):
    async def test_complex_id(self):
        v = out_python.Model__complex(r=1.0, i=2.0)
        async with get_db_sesh() as db_sesh:
            result = await out_python.complex_id(db_sesh, v)
        self.assertEqual(result.r, 1.0)
        self.assertEqual(result.i, 2.0)


class TestSQLAlchemy(unittest.IsolatedAsyncioTestCase): ...


async def run_nullables():
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        print(list(await out.nullables(session)))


async def run_count_leagues_by_nullable():
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        print(await out.count_leagues_by_nullable(session, "extra"))


async def run_get_mood():
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        print(await out.get_mood(session, "happy"))


async def run_complex_id():
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        v = out.Model__complex(r=1.0, i=2.0)
        print(repr(await out.complex_id(session, v)))
