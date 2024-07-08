import json
import unittest
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy import NullPool, literal, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from . import out_asyncpg_only, out_func, out_python


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


@asynccontextmanager
async def get_asyncpg_conn():
    conn = await asyncpg.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    yield conn
    await conn.close()


class TestAsyncpgOnly(unittest.IsolatedAsyncioTestCase):
    async def test_complex_id(self):
        v = out_asyncpg_only.Model__complex(r=1.0, i=2.0)
        async with get_asyncpg_conn() as conn:
            result = await out_asyncpg_only.complex_id(conn, v)
        assert result is not None
        self.assertEqual(result.r, 1.0)
        self.assertEqual(result.i, 2.0)

    async def test_jsonb_id(self):
        s = "test"
        async with get_asyncpg_conn() as conn:
            result = await out_asyncpg_only.jsonb_id(conn, s)
        self.assertEqual(result, s)

    async def test_set_of_complex_arrays(self):
        async with get_asyncpg_conn() as conn:
            result = await out_asyncpg_only.set_of_complex_arrays(conn)

        self.assertEqual(
            result,
            [
                [
                    out_asyncpg_only.Model__complex(r=1, i=2),
                    out_asyncpg_only.Model__complex(r=3, i=4),
                ],
                [
                    out_asyncpg_only.Model__complex(r=5, i=6),
                    out_asyncpg_only.Model__complex(r=7, i=8),
                ],
            ],
        )


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
        self.assertEqual(result, v)

    async def test_circle_id(self):
        c = asyncpg.Circle(center=asyncpg.Point(1, 2), radius=3)
        async with get_db_sesh() as db_sesh:
            result = await out_python.circle_id(db_sesh, c)
        assert isinstance(result, asyncpg.Circle)
        self.assertEqual(result, c)

    async def test_complex_array_id(self):
        ca = [out_python.Model__complex(r=2, i=4)]
        async with get_db_sesh() as db_sesh:
            result = await out_python.complex_array_id(db_sesh, ca)
        self.assertEqual(result, ca)

    async def test_set_of_complex_arrays(self):
        async with get_db_sesh() as db_sesh:
            result = await out_python.set_of_complex_arrays(db_sesh)
        self.assertEqual(
            list(result),
            [
                [
                    out_python.Model__complex(r=1, i=2),
                    out_python.Model__complex(r=3, i=4),
                ],
                [
                    out_python.Model__complex(r=5, i=6),
                    out_python.Model__complex(r=7, i=8),
                ],
            ],
        )

    async def test_unitthing(self):
        z = out_python.Model__complex(r=1, i=0)
        async with get_db_sesh() as db_sesh:
            result = await out_python.unitthing(db_sesh, z)
        self.assertEqual(result, z)

    async def test_get_mood(self):
        async with get_db_sesh() as db_sesh:
            result = await out_python.get_mood(
                db_sesh, out_python.Enum__mood.sad
            )
        self.assertEqual(result, out_python.Enum__mood.happy)

    async def test_jsonb_id(self):
        s = "test"
        async with get_db_sesh() as db_sesh:
            result = await out_python.jsonb_id(db_sesh, s)
        self.assertEqual(result, s)

    async def test_c2vector_id(self):
        c2v = out_python.Model__c2vector(
            z1=out_python.Model__complex(r=1, i=2),
            z2=out_python.Model__complex(r=3, i=4),
            moods=[out_python.Enum__mood.happy],
        )
        async with get_db_sesh() as db_sesh:
            result = await out_python.c2vector_id(db_sesh, c2v)
        self.assertEqual(result, c2v)

    @unittest.skip("See https://github.com/amacfie/sqlafunccodegen/issues/10")
    async def test_anyenum_f(self):
        e = out_python.Enum__mood.happy
        async with get_db_sesh() as db_sesh:
            result = await out_python.anyenum_f(db_sesh, e, [e])
        self.assertEqual(result, e)

    async def test_all_leagues(self):
        async with get_db_sesh() as db_sesh:
            result = await out_python.all_leagues(db_sesh)
        self.assertEqual(
            list(result),
            [
                out_python.Model__league(
                    id=1,
                    name="Premier League",
                    nullable=None,
                    stuff=None,
                    cs=None,
                ),
                out_python.Model__league(
                    id=2,
                    name="Bundesliga",
                    nullable="extra",
                    stuff=None,
                    cs=[
                        out_python.Model__complex(r=10, i=20),
                        out_python.Model__complex(r=30, i=40),
                    ],
                ),
            ],
        )


class TestSQLAlchemy(unittest.IsolatedAsyncioTestCase):
    async def test_complex_id(self):
        async with get_db_sesh() as db_sesh:
            result = (
                await db_sesh.execute(
                    select(out_func.complex_id(literal({"r": 1.0, "i": 2.0})))
                )
            ).scalar_one_or_none()
        assert result is not None
        self.assertEqual(result["r"], 1.0)
        self.assertEqual(result["i"], 2.0)
