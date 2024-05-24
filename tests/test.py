import asyncio
import importlib.util
import io
import pathlib
import sys
import tempfile
from typing import Any

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from sqlafunccodegen.main import main


out: Any = None

engine = create_async_engine(
    "postgresql+asyncpg://" + "postgres:postgres@localhost:5432/postgres",
    echo=True,
    poolclass=NullPool,
)


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


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        p = pathlib.Path(f.name)
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = captured_output
            main()
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        p.write_text(output)

        spec = importlib.util.spec_from_file_location("out", str(p))
        assert spec is not None
        out: Any = importlib.util.module_from_spec(spec)
        sys.modules["out"] = out
        assert spec.loader is not None
        spec.loader.exec_module(out)
        # the output is now loaded as a module named `out`

    asyncio.run(run_complex_id())
