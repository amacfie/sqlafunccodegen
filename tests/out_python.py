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
AnyArray = list[_T] | list['AnyArray'] | None
AnyArrayIn = Sequence[_T] | Sequence['AnyArray'] | None

def __convert_output(t, v):
    S = pydantic.create_model(
        'S',
        f=(t, ...),
        __config__=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )
    return S.model_validate({'f': v}).f  # type: ignore
ArrayIn__int4 = TypeAliasType('ArrayIn__int4', 'Sequence[Union[int, None]] | Sequence[ArrayIn__int4] | None')
ArrayIn__text = TypeAliasType('ArrayIn__text', 'Sequence[Union[str, None]] | Sequence[ArrayIn__text] | None')
Array__int4 = TypeAliasType('Array__int4', 'list[Union[int, None]] | list[Array__int4] | None')
Array__text = TypeAliasType('Array__text', 'list[Union[str, None]] | list[Array__text] | None')

class Enum__mood(str, Enum):
    happy = 'happy'
    sad = 'sad'
    neutral = 'neutral'

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
    list: 'Array__text'
async def array_id(
    db_sesh: AsyncSession, arr: ArrayIn__int4
) -> Array__int4:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'array_id')(sqlalchemy.literal(arr, type_=postgresql.ARRAY(postgresql.INTEGER)))
        )
    )).scalar_one_or_none()
    return __convert_output(Array__int4, r)

async def can_return_null(
    db_sesh: AsyncSession, 
) -> Union[str, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'can_return_null')()
        )
    )).scalar_one_or_none()
    return __convert_output(Union[str, None], r)

async def complex_id(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_id')(sqlalchemy.literal(None if z is None else z.model_dump(), type_=None))
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
            getattr(sqlalchemy.func, 'count_leagues_by_nullable')(sqlalchemy.literal(_nullable, type_=postgresql.TEXT))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[int, None], r)

async def do_anyrange(
    db_sesh: AsyncSession, r: Any
) -> Union[None, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'do_anyrange')(sqlalchemy.literal(r, type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[None, None], r)

async def get_lists(
    db_sesh: AsyncSession, _list: ArrayIn__text
) -> Iterable[Array__text]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_lists')(sqlalchemy.literal(_list, type_=postgresql.ARRAY(postgresql.TEXT)))
        )
    )).scalars()
    return (__convert_output(Array__text, i) for i in r)

async def get_mood(
    db_sesh: AsyncSession, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_mood')(sqlalchemy.literal(_mood, type_=postgresql.ENUM(name='mood')))
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

async def unitthing(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'unitthing')(sqlalchemy.literal(None if z is None else z.model_dump(), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__complex | None, r)
