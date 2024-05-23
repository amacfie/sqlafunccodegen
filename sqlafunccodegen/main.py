import asyncio
import dataclasses
import pathlib
from typing import Sequence

import dukpy
import typer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio.engine import AsyncEngine, create_async_engine


async def get_graphile_data(engine: AsyncEngine, schema: str) -> Sequence[dict]:
    version_q = select(func.current_setting("server_version_num"))
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        server_version_num = (await session.execute(version_q)).scalar_one()
    print("Postgres version", server_version_num)

    jsi = dukpy.JSInterpreter()
    jsi.loader.register_path(".")
    jsi.evaljs("m = require('introspectionQuery')")
    introspection_q = jsi.evaljs(
        "m.makeIntrospectionQuery(dukpy['serverVersionNum'])",
        serverVersionNum=server_version_num,
    )
    assert isinstance(introspection_q, str)

    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        conn = await session.connection()
        return (
            (await conn.exec_driver_sql(introspection_q, ([schema], False)))
            .scalars()
            .all()
        )


def main(
    outfile: pathlib.Path,
    dest: str = "postgres:postgres@localhost:5432/postgres",
    schema: str = "public",
):
    dest = dest.removeprefix("postgres://").removeprefix("postgresql://")
    python_generator = PythonGenerator(dest=dest, schema=schema)

    outfile.write_text(python_generator.generate())


@dataclasses.dataclass
class TypeStrings:
    python_type: str
    sqla_type: str | None
    python_type_anyenum: str | None = None


# types within a schema have unique names
pg_catalog_types = {
    "int4": TypeStrings(python_type="int", sqla_type="postgres.INTEGER"),
    "text": TypeStrings(python_type="str", sqla_type="postgres.TEXT"),
    "varchar": TypeStrings(python_type="str", sqla_type="postgres.VARCHAR"),
    "char": TypeStrings(python_type="str", sqla_type="postgres.CHAR"),
    "uuid": TypeStrings(python_type="UUID", sqla_type="sqlalchemy.UUID"),
    "bool": TypeStrings(python_type="bool", sqla_type="sqlalchemy.Boolean"),
    "void": TypeStrings(python_type="None", sqla_type=None),
    "json": TypeStrings(python_type="JsonValue", sqla_type="postgresql.JSON"),
    "jsonb": TypeStrings(python_type="JsonValue", sqla_type="postgresql.JSONB"),
    "anyarray": TypeStrings(
        python_type="AnyArray[_T]",
        sqla_type="None",
        python_type_anyenum="AnyArray[_E]",
    ),
    "anyelement": TypeStrings(
        python_type="_T", sqla_type="None", python_type_anyenum="_E"
    ),
    "anyenum": TypeStrings(
        python_type="_E", sqla_type="None", python_type_anyenum="_E"
    ),
    "anycompatible": TypeStrings(
        python_type="_T", sqla_type="None", python_type_anyenum="_E"
    ),
    "anycompatiblearray": TypeStrings(
        python_type="AnyArray[_T]",
        sqla_type="None",
        python_type_anyenum="AnyArray[_E]",
    ),
    "bit": TypeStrings(
        python_type="asyncpg.BitString", sqla_type="postgresql.BIT"
    ),
    "box": TypeStrings(python_type="asyncpg.Box", sqla_type="None"),
    "bytea": TypeStrings(python_type="bytes", sqla_type="postgresql.BYTEA"),
    "cidr": TypeStrings(
        python_type="IPv4Network | IPv6Network", sqla_type="postgresql.CIDR"
    ),
    "inet": TypeStrings(
        python_type="IPv4Interface | IPv6Interface | IPv4Address | IPv6Address",
        sqla_type="postgresql.CIDR",
    ),
    "macaddr": TypeStrings(python_type="str", sqla_type="postgresql.MACADDR"),
    "date": TypeStrings(
        python_type="datetime.date", sqla_type="postgresql.DATE"
    ),
    "time": TypeStrings(
        python_type="datetime.time", sqla_type="postgresql.TIME(timezone=False)"
    ),
    "timetz": TypeStrings(
        python_type="datetime.time", sqla_type="postgresql.TIME(timezone=True)"
    ),
    "timestamp": TypeStrings(
        python_type="datetime.datetime",
        sqla_type="postgresql.TIMESTAMP(timezone=False)",
    ),
    "timestamptz": TypeStrings(
        python_type="datetime.datetime",
        sqla_type="postgresql.TIMESTAMP(timezone=True)",
    ),
    "interval": TypeStrings(
        python_type="datetime.timedelta",
        sqla_type="postgresql.INTERVAL",
    ),
    "float": TypeStrings(python_type="float", sqla_type="postgresql.FLOAT"),
    "bigint": TypeStrings(python_type="int", sqla_type="postgresql.BIGINT"),
    "numeric": TypeStrings(
        python_type="Decimal", sqla_type="postgresql.NUMERIC"
    ),
    "money": TypeStrings(python_type="str", sqla_type="postgresql.MONEY"),
}

FRONTMATTER = """import datetime
from decimal import Decimal
from enum import Enum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Any, Iterable, Literal, TypeVar, Union
from uuid import UUID

import asyncpg
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession


JsonValue = None | bool | float | int | str | list['JsonValue'] | dict[str, 'JsonValue']
_T = TypeVar('_T')
_E = TypeVar('_E', bound=Enum)
AnyArray = list[_T] | list['AnyArray'] | None
"""


class PythonGenerator:
    out_recursive_type_defs: set[str] = set()
    out_enums: set[str] = set()

    def __init__(self, dest: str, schema: str):
        self.schema = schema

        engine = create_async_engine("postgresql+asyncpg://" + dest)

        self.graphile_data = asyncio.run(
            get_graphile_data(engine=engine, schema=schema)
        )

        self.graphile_type_by_id = {
            obj["id"]: obj
            for obj in self.graphile_data
            if obj["kind"] == "type"
        }

    def graphile_type_to_python(
        self, graphile_type: dict, anyenum: bool
    ) -> str:
        if graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                if (
                    anyenum
                    and pg_catalog_types[
                        graphile_type["name"]
                    ].python_type_anyenum
                ):
                    pct = pg_catalog_types[
                        graphile_type["name"]
                    ].python_type_anyenum
                else:
                    pct = pg_catalog_types[graphile_type["name"]].python_type

                # we have to avoid `None | None` because it's an error
                return f"Union[{pct}, None]"
            elif graphile_type["isPgArray"]:
                item_graphile_type = self.graphile_type_by_id[
                    graphile_type["arrayItemTypeId"]
                ]
                item_python_type = self.graphile_type_to_python(
                    item_graphile_type,
                    anyenum=anyenum,
                )
                ret = f"Array__{item_graphile_type['name']}"
                rtd = (
                    f"{ret} = list['{item_python_type}'] | list['{ret}'] | None"
                )
                self.out_recursive_type_defs.add(rtd)
                return ret
        elif graphile_type["namespaceName"] == self.schema:
            if graphile_type["enumVariants"]:
                enum_name = f"Enum__{graphile_type['name']}"
                e = f"class {enum_name}(str, Enum):\n"
                e += "\n".join(
                    f"    {ev} = '{ev}'" for ev in graphile_type["enumVariants"]
                )
                self.out_enums.add(e)
                return f"{enum_name} | None"
        return "Any"

    def graphile_type_to_sqla(self, graphile_type: dict) -> str:
        if graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                sqla_type = pg_catalog_types[graphile_type["name"]].sqla_type
                assert sqla_type is not None
                return sqla_type
            elif graphile_type["isPgArray"]:
                item_graphile_type = self.graphile_type_by_id[
                    graphile_type["arrayItemTypeId"]
                ]
                item_sqla_type = self.graphile_type_to_sqla(item_graphile_type)
                return f"postgresql.ARRAY({item_sqla_type})"
        elif graphile_type["namespaceName"] == self.schema:
            if graphile_type["enumVariants"]:
                # we don't need the *enums parameter because we just use this
                # for casting
                return (
                    "postgresql.ENUM(name=" f"{repr(graphile_type['name'])}" ")"
                )
        return "None"

    def generate(self):
        procedures = [
            gd for gd in self.graphile_data if gd["kind"] == "procedure"
        ]
        procedure_names = [procedure["name"] for procedure in procedures]
        overloads = {
            procedure["name"]
            for procedure in procedures
            if procedure_names.count(procedure["name"]) > 1
        }
        generated_procedures = [
            self.generate_procedure(
                procedure, overload=procedure["name"] in overloads
            )
            for procedure in procedures
        ]
        out_procedures = "\n\n".join(
            gp for gp in generated_procedures if gp is not None
        )

        ret = FRONTMATTER
        ret += "\n".join(self.out_recursive_type_defs) + "\n\n"
        ret += "\n".join(self.out_enums) + "\n\n"
        ret += out_procedures

        return ret

    def generate_procedure(self, procedure: dict, overload: bool) -> str | None:
        print(procedure)
        return_type = self.graphile_type_by_id[procedure["returnTypeId"]]
        print(return_type, "\n")

        # if any arg mode is not IN, we don't handle it
        # variadic params aren't supported in Graphile
        if any(am != "i" for am in procedure["argModes"]):
            return None

        arg_types = [
            self.graphile_type_by_id[arg_id]
            for arg_id in procedure["argTypeIds"]
        ]

        anyenum = "anyenum" in (
            {at["name"] for at in arg_types} | {return_type["name"]}
        )

        scalar_return_type = self.graphile_type_to_python(
            return_type, anyenum=anyenum
        )
        if procedure["returnsSet"]:
            python_return_type = f"Iterable[{scalar_return_type}]"
        else:
            python_return_type = scalar_return_type

        out_params = ", ".join(
            arg_name
            + ": "
            + self.graphile_type_to_python(arg_type, anyenum=anyenum)
            for arg_type, arg_name in zip(arg_types, procedure["argNames"])
        )

        # e.g. if a function takes a JSONB, we can't just pass a python string
        # because that becomes `TEXT` in pg which is a type error
        out_args = ", ".join(
            f"sqlalchemy.bindparam(key=None, value={arg_name},"
            f" type_={self.graphile_type_to_sqla(arg_type)})"
            for arg_type, arg_name in zip(arg_types, procedure["argNames"])
        )

        if procedure["returnsSet"]:
            out_method = "scalars"
        else:
            out_method = "scalar_one_or_none"

        if procedure["description"] is None:
            out_docstring = ""
        else:
            out_docstring = repr(procedure["description"])

        # an alternative would be using NewType to differentiate between
        # ambiguous Python types, e.g. Money = NewType("Money", str)
        # and then use @typing.overload
        if overload:
            out_name = (
                procedure["name"]
                + "__"
                + "__".join(at["name"] for at in arg_types)
            )
        else:
            out_name = procedure["name"]

        return f"""async def {out_name}(
    db_sesh: AsyncSession, {out_params}
) -> {python_return_type}:
    {out_docstring}
    return (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, '{procedure["name"]}')({out_args})
        )
    )).{out_method}()"""


def cli():
    typer.run(main)
    return 0


if __name__ == "__main__":
    cli()
