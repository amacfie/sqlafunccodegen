import unittest
from contextlib import asynccontextmanager

from sqlalchemy import NullPool, literal, select
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
        assert result is not None
        self.assertEqual(result.r, 1.0)
        self.assertEqual(result.i, 2.0)

    async def test_array_id(self):
        v = [[1, 2], [3, 4]]
        async with get_db_sesh() as db_sesh:
            result = await out_python.array_id(db_sesh, v)
        assert result == v


class TestSQLAlchemy(unittest.IsolatedAsyncioTestCase):
    async def test_complex_id(self):
        async with get_db_sesh() as db_sesh:
            result = (
                await db_sesh.execute(
                    select(
                        out_sqlalchemy.complex_id(literal({"r": 1.0, "i": 2.0}))
                    )
                )
            ).scalar_one_or_none()
        assert result is not None
        self.assertEqual(result["r"], 1.0)
        self.assertEqual(result["i"], 2.0)
