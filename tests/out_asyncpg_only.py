import datetime
from decimal import Decimal
from enum import Enum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Annotated, Any, Iterable, Mapping, Sequence, TypeVar, Union
from typing_extensions import TypeAliasType
from uuid import UUID

import asyncpg
import pydantic
_T = TypeVar('_T')
_E = TypeVar('_E', bound=Enum)
AnyArray = list[_T] | list['AnyArray']
AnyArrayIn = Sequence[_T] | Sequence['AnyArray']
JsonFrozen = Union[Mapping[str, "JsonFrozen"], Sequence["JsonFrozen"], str, int, float, bool, None]

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
ArrayIn__complex = TypeAliasType('ArrayIn__complex', 'Sequence[Model__complex | None] | Sequence[ArrayIn__complex] | None')
ArrayIn__int4 = TypeAliasType('ArrayIn__int4', 'Sequence[Union[int, None]] | Sequence[ArrayIn__int4] | None')
ArrayIn__text = TypeAliasType('ArrayIn__text', 'Sequence[Union[str, None]] | Sequence[ArrayIn__text] | None')
Array__complex = TypeAliasType('Array__complex', 'list[Model__complex | None] | list[Array__complex] | None')
Array__int4 = TypeAliasType('Array__int4', 'list[Union[int, None]] | list[Array__int4] | None')
Array__mood = TypeAliasType('Array__mood', 'list[Enum__mood | None] | list[Array__mood] | None')
Array__text = TypeAliasType('Array__text', 'list[Union[str, None]] | list[Array__text] | None')

class Enum__mood(str, Enum):
    happy = 'happy'
    sad = 'sad'
    neutral = 'neutral'

class Model__c2vector(pydantic.BaseModel):

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
    z1: 'Model__complex | None'
    z2: 'Model__complex | None'
    moods: 'Array__mood'


class Model__complex(pydantic.BaseModel):
    'A complex number'

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
    r: Annotated['Union[float, None]', pydantic.Field(description='The real part')]
    i: 'Union[float, None]'


class Model__league(pydantic.BaseModel):

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
    id: 'Union[int, None]'
    name: 'Union[str, None]'
    nullable: 'Union[str, None]'
    stuff: 'Array__text'
    cs: 'Array__complex'
async def all_leagues(
    conn: asyncpg.Connection, 
) -> Iterable[Model__league | None]:
    
    return [__convert_output(Model__league | None, r[0]) for r in await conn.fetch('select all_leagues()', )]

async def array_id(
    conn: asyncpg.Connection, arr: ArrayIn__int4
) -> Array__int4:
    
    return __convert_output(Array__int4, await conn.fetchval('select array_id($1)', __convert_input(arr)))

async def c2vector_id(
    conn: asyncpg.Connection, c: Model__c2vector | None
) -> Model__c2vector | None:
    
    return __convert_output(Model__c2vector | None, await conn.fetchval('select c2vector_id($1)', __convert_input(c)))

async def can_return_null(
    conn: asyncpg.Connection, 
) -> Union[str, None]:
    
    return __convert_output(Union[str, None], await conn.fetchval('select can_return_null()', ))

async def circle_id(
    conn: asyncpg.Connection, c: Union[asyncpg.Circle, None]
) -> Union[asyncpg.Circle, None]:
    
    return __convert_output(Union[asyncpg.Circle, None], await conn.fetchval('select circle_id($1)', __convert_input(c)))

async def complex_array_id(
    conn: asyncpg.Connection, ca: ArrayIn__complex
) -> Array__complex:
    
    return __convert_output(Array__complex, await conn.fetchval('select complex_array_id($1)', __convert_input(ca)))

async def complex_id(
    conn: asyncpg.Connection, z: Model__complex | None
) -> Model__complex | None:
    
    return __convert_output(Model__complex | None, await conn.fetchval('select complex_id($1)', __convert_input(z)))

async def count_leagues(
    conn: asyncpg.Connection, 
) -> Union[pydantic.JsonValue, None]:
    
    return __convert_output(Union[pydantic.JsonValue, None], await conn.fetchval('select count_leagues()', ))

async def count_leagues_by_nullable(
    conn: asyncpg.Connection, _nullable: Union[str, None]
) -> Union[int, None]:
    'Count leagues by nullable'
    return __convert_output(Union[int, None], await conn.fetchval('select count_leagues_by_nullable($1)', __convert_input(_nullable)))

async def get_mood(
    conn: asyncpg.Connection, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    return __convert_output(Enum__mood | None, await conn.fetchval('select get_mood($1)', __convert_input(_mood)))

async def get_range(
    conn: asyncpg.Connection, 
) -> Any:
    
    return __convert_output(Any, await conn.fetchval('select get_range()', ))

async def get_stuff(
    conn: asyncpg.Connection, _stuff: ArrayIn__text
) -> Iterable[Array__text]:
    
    return [__convert_output(Array__text, r[0]) for r in await conn.fetch('select get_stuff($1)', __convert_input(_stuff))]

async def getall(
    conn: asyncpg.Connection, 
) -> Iterable[Model__league | None]:
    
    return [__convert_output(Model__league | None, r[0]) for r in await conn.fetch('select getall()', )]

async def ids(
    conn: asyncpg.Connection, 
) -> Iterable[Union[int, None]]:
    
    return [__convert_output(Union[int, None], r[0]) for r in await conn.fetch('select ids()', )]

async def jsonb_id(
    conn: asyncpg.Connection, j: Union[JsonFrozen, None]
) -> Union[pydantic.JsonValue, None]:
    
    return __convert_output(Union[pydantic.JsonValue, None], await conn.fetchval('select jsonb_id($1)', __convert_input(j)))

async def nullables(
    conn: asyncpg.Connection, 
) -> Iterable[Union[str, None]]:
    
    return [__convert_output(Union[str, None], r[0]) for r in await conn.fetch('select nullables()', )]

async def retvoid(
    conn: asyncpg.Connection, 
) -> Union[None, None]:
    
    return __convert_output(Union[None, None], await conn.fetchval('select retvoid()', ))

async def set_of_complex_arrays(
    conn: asyncpg.Connection, 
) -> Iterable[Array__complex]:
    
    return [__convert_output(Array__complex, r[0]) for r in await conn.fetch('select set_of_complex_arrays()', )]

async def unitthing(
    conn: asyncpg.Connection, z: Model__complex | None
) -> Model__complex | None:
    
    return __convert_output(Model__complex | None, await conn.fetchval('select unitthing($1)', __convert_input(z)))
