import asyncio
import pathlib
from collections import Counter, defaultdict
from enum import Enum
from functools import lru_cache
from typing import Annotated, Sequence

import dukpy
import typer
from pydantic import BaseModel, model_validator
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

    jsi = dukpy.JSInterpreter()
    pkg_path = pathlib.Path(__file__).parent.resolve()
    jsi.loader.register_path(str(pkg_path))
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


class Mode(str, Enum):
    python = "python"
    sqlalchemy = "sqlalchemy"


def main(
    dest: Annotated[
        str,
        typer.Option(help="user:pass@host:port/dbname"),
    ] = "postgres:postgres@localhost:5432/postgres",
    schema: str = "public",
    mode: Annotated[
        Mode,
        typer.Option(
            help=(
                "'python' mode generates functions that take and return Python"
                " values, while 'sqlalchemy' mode generates functions that act"
                " as sqlalchemy.func functions and can be used within a"
                " SQLAlchemy SQL expression."
            )
        ),
    ] = Mode.python,
):
    dest = dest.removeprefix("postgres://").removeprefix("postgresql://")
    python_generator = PythonGenerator(dest=dest, schema=schema)
    print(python_generator.generate(mode))


class TypeStrings(BaseModel):
    python_type: str
    # if an anyenum type is used, all polymorphic types become enums
    python_type_anyenum: str | None = None
    python_type_in: str | None = None
    python_type_anyenum_in: str | None = None
    sqla_type: str | None = None

    @model_validator(mode="after")
    def validate_model(self):
        if self.python_type_anyenum_in is None:
            assert (self.python_type_in is None) or (
                self.python_type_anyenum is None
            )
        return self


# types within a schema have unique names
# https://magicstack.github.io/asyncpg/current/usage.html#type-conversion
pg_catalog_types = {
    "int2": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "int4": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "int8": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "text": TypeStrings(python_type="str", sqla_type="postgresql.TEXT"),
    "varchar": TypeStrings(python_type="str", sqla_type="postgresql.VARCHAR"),
    "char": TypeStrings(python_type="str", sqla_type="postgresql.CHAR"),
    "uuid": TypeStrings(python_type="UUID", sqla_type="sqlalchemy.UUID"),
    "bool": TypeStrings(python_type="bool", sqla_type="sqlalchemy.Boolean"),
    "void": TypeStrings(python_type="None"),
    "json": TypeStrings(
        python_type="pydantic.JsonValue", sqla_type="postgresql.JSON"
    ),
    "jsonb": TypeStrings(
        python_type="pydantic.JsonValue", sqla_type="postgresql.JSONB"
    ),
    "anyarray": TypeStrings(
        python_type="AnyArray[_T]",
        sqla_type="None",
        python_type_anyenum="AnyArray[_E]",
        python_type_in="AnyArrayIn[_T]",
        python_type_anyenum_in="AnyArrayIn[_E]",
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
        python_type_in="AnyArrayIn[_T]",
        python_type_anyenum_in="AnyArrayIn[_E]",
    ),
    "bit": TypeStrings(
        python_type="asyncpg.BitString", sqla_type="postgresql.BIT"
    ),
    "varbit": TypeStrings(
        python_type="asyncpg.BitString", sqla_type="postgresql.VARBINARY"
    ),
    "box": TypeStrings(python_type="asyncpg.Box", sqla_type="None"),
    "bytea": TypeStrings(python_type="bytes", sqla_type="postgresql.BYTEA"),
    "cidr": TypeStrings(
        python_type="IPv4Network | IPv6Network", sqla_type="postgresql.CIDR"
    ),
    "inet": TypeStrings(
        python_type="IPv4Interface | IPv6Interface | IPv4Address | IPv6Address",
        sqla_type="postgresql.INET",
    ),
    "macaddr": TypeStrings(python_type="str", sqla_type="postgresql.MACADDR"),
    "circle": TypeStrings(python_type="asyncpg.Circle", sqla_type="None"),
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
    "line": TypeStrings(python_type="asyncpg.Line", sqla_type="None"),
    "lseg": TypeStrings(python_type="asyncpg.LineSegment", sqla_type="None"),
    "path": TypeStrings(python_type="asyncpg.Path", sqla_type="None"),
    "point": TypeStrings(python_type="asyncpg.Point", sqla_type="None"),
    "polygon": TypeStrings(python_type="asyncpg.Polygon", sqla_type="None"),
    "float4": TypeStrings(
        python_type="float", sqla_type="postgresql.FLOAT(24)"
    ),
    "float8": TypeStrings(
        python_type="float", sqla_type="postgresql.FLOAT(53)"
    ),
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
from typing import Annotated, Any, Iterable, Sequence, TypeVar, Union
from typing_extensions import TypeAliasType
from uuid import UUID

import asyncpg
import pydantic
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession


_T = TypeVar('_T')
_E = TypeVar('_E', bound=Enum)
AnyArray = list[_T] | list['AnyArray']
AnyArrayIn = Sequence[_T] | Sequence['AnyArray']

def __convert_output(t, v):
    S = pydantic.create_model(
        'S',
        f=(t, ...),
        __config__=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )
    return S.model_validate({'f': v}).f  # type: ignore


def __convert_input(v):
    class S(pydantic.BaseModel):
        model_config=pydantic.ConfigDict(arbitrary_types_allowed=True)
        f: Any

    return S(f=v).model_dump()["f"]  # type: ignore
"""


class PythonGenerator:
    # arrays in Postgres can have any dimension so we use recursive type aliases
    out_recursive_type_defs: set[str] = set()
    out_enums: set[str] = set()
    class_ids_to_generate: set[str] = set()

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
        self.graphile_class_by_id = {
            obj["id"]: obj
            for obj in self.graphile_data
            if obj["kind"] == "class"
        }
        self.graphile_attrs_by_class_id = defaultdict(list)
        for obj in self.graphile_data:
            if obj["kind"] == "attribute":
                self.graphile_attrs_by_class_id[obj["classId"]].append(obj)
        for key in self.graphile_attrs_by_class_id:
            self.graphile_attrs_by_class_id[key].sort(key=lambda x: x["num"])

    @lru_cache(maxsize=None)
    def graphile_type_to_python(
        self, graphile_type_id: str, anyenum: bool, in_: bool | None = None
    ) -> str:
        graphile_type = self.graphile_type_by_id[graphile_type_id]

        if graphile_type["isPgArray"]:
            item_graphile_type = self.graphile_type_by_id[
                graphile_type["arrayItemTypeId"]
            ]
            if in_:
                ret = f"ArrayIn__{item_graphile_type['name']}"
            else:
                ret = f"Array__{item_graphile_type['name']}"
            item_python_type = self.graphile_type_to_python(
                item_graphile_type["id"],
                anyenum=anyenum,
                in_=in_,
            )
            # https://github.com/pydantic/pydantic/issues/8346
            if in_:
                rtd = (
                    f"{ret} = TypeAliasType('{ret}',"
                    f" 'Sequence[{item_python_type}] | Sequence[{ret}]"
                    " | None')"
                )
            else:
                rtd = (
                    f"{ret} = TypeAliasType('{ret}',"
                    f" 'list[{item_python_type}] | list[{ret}] | None'"
                    ")"
                )
            self.out_recursive_type_defs.add(rtd)
            return ret
        elif graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                ts = pg_catalog_types[graphile_type["name"]]
                if anyenum and in_ and ts.python_type_anyenum_in:
                    pt = ts.python_type_anyenum_in
                elif anyenum and ts.python_type_anyenum:
                    pt = ts.python_type_anyenum
                elif in_ and ts.python_type_in:
                    pt = ts.python_type_in
                else:
                    pt = ts.python_type

                # we have to avoid `None | None` because it's an error
                return f"Union[{pt}, None]"
        elif graphile_type["namespaceName"] == self.schema:
            if graphile_type["enumVariants"]:
                enum_name = f"Enum__{graphile_type['name']}"
                e = f"class {enum_name}(str, Enum):\n"
                if graphile_type["description"]:
                    e += f"    {repr(graphile_type['description'])}\n"
                e += "\n".join(
                    f"    {ev} = '{ev}'" for ev in graphile_type["enumVariants"]
                )
                self.out_enums.add(e)
                return f"{enum_name} | None"
            elif graphile_type["classId"] is not None:
                self.class_ids_to_generate.add(graphile_type["classId"])
                return f"Model__{graphile_type['name']} | None"
            elif graphile_type["domainBaseTypeId"] is not None:
                return self.graphile_type_to_python(
                    graphile_type["domainBaseTypeId"],
                    anyenum=anyenum,
                    in_=in_,
                )
        return "Any"

    @lru_cache(maxsize=None)
    def graphile_type_to_sqla(self, graphile_type_id: str) -> str:
        graphile_type = self.graphile_type_by_id[graphile_type_id]

        if graphile_type["isPgArray"]:
            item_graphile_type = self.graphile_type_by_id[
                graphile_type["arrayItemTypeId"]
            ]
            item_sqla_type = self.graphile_type_to_sqla(
                item_graphile_type["id"]
            )
            if item_sqla_type == "None":
                # ARRAY(None) doesn't seem to work
                return "None"
            else:
                return f"postgresql.ARRAY({item_sqla_type})"
        elif graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                sqla_type = pg_catalog_types[graphile_type["name"]].sqla_type
                assert sqla_type is not None
                return sqla_type
        elif graphile_type["namespaceName"] == self.schema:
            if graphile_type["enumVariants"]:
                # we don't need the *enums parameter because we just use this
                # for casting
                return (
                    "postgresql.ENUM(name=" f"{repr(graphile_type['name'])}" ")"
                )
            elif graphile_type["domainBaseTypeId"] is not None:
                return self.graphile_type_to_sqla(
                    graphile_type["domainBaseTypeId"],
                )
        return "None"

    def generate(self, mode: Mode) -> str:
        procedures = [
            gd for gd in self.graphile_data if gd["kind"] == "procedure"
        ]

        # Graphile doesn't seem to support overloads
        name_count = Counter(procedure["name"] for procedure in procedures)
        overloads = {
            procedure["name"]
            for procedure in procedures
            if name_count[procedure["name"]] > 1
        }
        assert not overloads

        generated_procedures = [
            self.generate_procedure(procedure, mode=mode)
            for procedure in procedures
        ]
        out_procedures = "\n\n".join(
            sorted(gp for gp in generated_procedures if gp is not None)
        )

        out_models = "\n\n".join(sorted(self.generate_models()))

        ret = FRONTMATTER
        if mode == Mode.python:
            ret += "\n".join(sorted(self.out_recursive_type_defs)) + "\n\n"
            ret += "\n".join(sorted(self.out_enums)) + "\n\n"
            ret += out_models
        ret += out_procedures

        return ret

    def generate_models(self) -> list[str]:
        completed_class_ids = set()
        ret = []
        # items can get added to the set while we're iterating
        while self.class_ids_to_generate:
            class_id = self.class_ids_to_generate.pop()
            if class_id in completed_class_ids:
                continue
            class_ = self.graphile_class_by_id[class_id]
            out = f"class Model__{class_['name']}(pydantic.BaseModel):\n"
            graphile_type = self.graphile_type_by_id[class_["typeId"]]
            if graphile_type["description"]:
                out += f"    {repr(graphile_type['description'])}\n"
            out += """
    model_config=pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_model(cls, data):
        if isinstance(data, asyncpg.Record):
            return dict(data.items())
        elif isinstance(data, tuple):
            # not sure when this can happen
            return dict(
                (k, v)
                for k, v in zip(cls.model_fields, data)
            )
        else:
            return data
"""

            attrs = self.graphile_attrs_by_class_id[class_id]
            for attr in attrs:
                basic_attr_type = (
                    "'"
                    + self.graphile_type_to_python(
                        # composite types can't have polymorphic types in them,
                        # so the anyenum value doesn't matter
                        attr["typeId"],
                        anyenum=False,
                    )
                    + "'"
                )
                if attr["description"]:
                    attr_type = (
                        f"Annotated[{basic_attr_type}, pydantic.Field("
                        f"description={repr(attr['description'])}"
                        ")]"
                    )
                else:
                    attr_type = basic_attr_type
                out += f"    {attr['name']}: {attr_type}\n"

            ret.append(out)
            completed_class_ids.add(class_id)
        return ret

    def generate_procedure(self, procedure: dict, mode: Mode) -> str | None:
        # if any arg mode is not IN, we don't handle it
        # variadic params aren't supported in Graphile
        if any(am != "i" for am in procedure["argModes"]):
            return None

        return_type = self.graphile_type_by_id[procedure["returnTypeId"]]
        arg_types = [
            self.graphile_type_by_id[arg_id]
            for arg_id in procedure["argTypeIds"]
        ]

        anyenum = "anyenum" in (
            {at["name"] for at in arg_types} | {return_type["name"]}
        )

        scalar_return_type = self.graphile_type_to_python(
            return_type["id"], anyenum=anyenum, in_=False
        )
        if procedure["returnsSet"]:
            # we won't get null
            out_return_type = f"Iterable[{scalar_return_type}]"
        else:
            out_return_type = scalar_return_type

        if procedure["returnsSet"]:
            out_python_return_stmt = (
                f"return ("
                f"__convert_output({scalar_return_type}, i)"
                f" for i in r)"
            )
        else:
            out_python_return_stmt = (
                f"return __convert_output({scalar_return_type}, r)"
            )

        params_list = []
        for arg_type, arg_name in zip(arg_types, procedure["argNames"]):
            s = f"{arg_name}: "
            if mode == "python":
                s += self.graphile_type_to_python(
                    arg_type["id"], anyenum=anyenum, in_=True
                )
            else:
                s += "Any"
            params_list.append(s)
        out_params = ", ".join(params_list)

        if mode == "python":
            out_args_list = []
            for arg_type, arg_name in zip(arg_types, procedure["argNames"]):
                v = f"__convert_input({arg_name})"
                # e.g. if a function takes a JSONB, we can't just pass a python
                # string because that becomes `TEXT` in pg which is a type
                # error. however, in some circumstances the call works if type_
                # is None.
                sqla_type = self.graphile_type_to_sqla(arg_type["id"])
                out_args_list.append(
                    f"sqlalchemy.literal({v}, type_={sqla_type})"
                )
            out_args = ", ".join(out_args_list)
        else:
            out_args = ", ".join(arg_name for arg_name in procedure["argNames"])

        if procedure["returnsSet"]:
            # each row has only one column, which may be a composite type
            out_method = "scalars"
        else:
            # we use _or_none just in case
            out_method = "scalar_one_or_none"

        if procedure["description"] is None:
            out_docstring = ""
        else:
            out_docstring = repr(procedure["description"])

        if mode == "python":
            return f"""async def {procedure['name']}(
    db_sesh: AsyncSession, {out_params}
) -> {out_return_type}:
    {out_docstring}
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, '{procedure["name"]}')({out_args})
        )
    )).{out_method}()
    {out_python_return_stmt}"""
        else:
            return f"""def {procedure['name']}(
    {out_params}
) -> Any:
    {out_docstring}
    return getattr(sqlalchemy.func, '{procedure["name"]}')({out_args})"""


def cli() -> int:
    typer.run(main)
    return 0


if __name__ == "__main__":
    exit(cli())
