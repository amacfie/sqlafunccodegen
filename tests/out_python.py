import datetime
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
    'A complex number'
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
    db_sesh: AsyncSession, 
) -> Iterable[Model__league | None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'all_leagues')()
        )
    )).scalars()
    return (__convert_output(Model__league | None, i) for i in r)

async def anyenum_f(
    db_sesh: AsyncSession, a: Union[_E, None], b: Union[AnyArrayIn[_E], None]
) -> Union[_E, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'anyenum_f')(sqlalchemy.literal(__convert_input(a), type_=None), sqlalchemy.literal(__convert_input(b), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[_E, None], r)

async def array_id(
    db_sesh: AsyncSession, arr: ArrayIn__int4
) -> Array__int4:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'array_id')(sqlalchemy.literal(__convert_input(arr), type_=postgresql.ARRAY(postgresql.INTEGER)))
        )
    )).scalar_one_or_none()
    return __convert_output(Array__int4, r)

async def c2vector_id(
    db_sesh: AsyncSession, c: Model__c2vector | None
) -> Model__c2vector | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'c2vector_id')(sqlalchemy.literal(__convert_input(c), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__c2vector | None, r)

async def can_return_null(
    db_sesh: AsyncSession, 
) -> Union[str, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'can_return_null')()
        )
    )).scalar_one_or_none()
    return __convert_output(Union[str, None], r)

async def circle_id(
    db_sesh: AsyncSession, c: Union[asyncpg.Circle, None]
) -> Union[asyncpg.Circle, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'circle_id')(sqlalchemy.literal(__convert_input(c), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[asyncpg.Circle, None], r)

async def complex_array_id(
    db_sesh: AsyncSession, ca: ArrayIn__complex
) -> Array__complex:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_array_id')(sqlalchemy.literal(__convert_input(ca), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Array__complex, r)

async def complex_id(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_id')(sqlalchemy.literal(__convert_input(z), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__complex | None, r)

async def count_leagues(
    db_sesh: AsyncSession, 
) -> Union[pydantic.JsonValue, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'count_leagues')()
        )
    )).scalar_one_or_none()
    return __convert_output(Union[pydantic.JsonValue, None], r)

async def count_leagues_by_nullable(
    db_sesh: AsyncSession, _nullable: Union[str, None]
) -> Union[int, None]:
    'Count leagues by nullable'
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'count_leagues_by_nullable')(sqlalchemy.literal(__convert_input(_nullable), type_=postgresql.TEXT))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[int, None], r)

async def do_anyrange(
    db_sesh: AsyncSession, r: Any
) -> Union[None, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'do_anyrange')(sqlalchemy.literal(__convert_input(r), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[None, None], r)

async def first_any(
    db_sesh: AsyncSession, a: Union[_T, None], b: Union[AnyArrayIn[_T], None]
) -> Union[_T, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'first_any')(sqlalchemy.literal(__convert_input(a), type_=None), sqlalchemy.literal(__convert_input(b), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[_T, None], r)

async def get_mood(
    db_sesh: AsyncSession, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_mood')(sqlalchemy.literal(__convert_input(_mood), type_=postgresql.ENUM(name='mood')))
        )
    )).scalar_one_or_none()
    return __convert_output(Enum__mood | None, r)

async def get_range(
    db_sesh: AsyncSession, 
) -> Any:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_range')()
        )
    )).scalar_one_or_none()
    return __convert_output(Any, r)

async def get_stuff(
    db_sesh: AsyncSession, _stuff: ArrayIn__text
) -> Iterable[Array__text]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_stuff')(sqlalchemy.literal(__convert_input(_stuff), type_=postgresql.ARRAY(postgresql.TEXT)))
        )
    )).scalars()
    return (__convert_output(Array__text, i) for i in r)

async def getall(
    db_sesh: AsyncSession, 
) -> Iterable[Model__league | None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'getall')()
        )
    )).scalars()
    return (__convert_output(Model__league | None, i) for i in r)

async def ids(
    db_sesh: AsyncSession, 
) -> Iterable[Union[int, None]]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'ids')()
        )
    )).scalars()
    return (__convert_output(Union[int, None], i) for i in r)

async def jsonb_id(
    db_sesh: AsyncSession, j: Union[pydantic.JsonValue, None]
) -> Union[pydantic.JsonValue, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'jsonb_id')(sqlalchemy.literal(__convert_input(j), type_=postgresql.JSONB))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[pydantic.JsonValue, None], r)

async def nullables(
    db_sesh: AsyncSession, 
) -> Iterable[Union[str, None]]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'nullables')()
        )
    )).scalars()
    return (__convert_output(Union[str, None], i) for i in r)

async def retvoid(
    db_sesh: AsyncSession, 
) -> Union[None, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'retvoid')()
        )
    )).scalar_one_or_none()
    return __convert_output(Union[None, None], r)

async def set_of_complex_arrays(
    db_sesh: AsyncSession, 
) -> Iterable[Array__complex]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'set_of_complex_arrays')()
        )
    )).scalars()
    return (__convert_output(Array__complex, i) for i in r)

async def unitthing(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'unitthing')(sqlalchemy.literal(__convert_input(z), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__complex | None, r)
