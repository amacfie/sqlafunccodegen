import datetime
from decimal import Decimal
from enum import Enum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Annotated, Any, Iterable, TypeVar, Union
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
Array__text = TypeAliasType('Array__text', 'list[Union[str, None]] | list[Array__text] | None')

class Enum__mood(str, Enum):
    happy = 'happy'
    sad = 'sad'
    neutral = 'neutral'

class Model__complex(pydantic.BaseModel):
    'A complex number'
    r: Annotated['Union[float, None]', pydantic.Field(description='The real part')]
    i: 'Union[float, None]'


class Model__league(pydantic.BaseModel):
    id: 'Union[int, None]'
    name: 'Union[str, None]'
    nullable: 'Union[str, None]'
    list: 'Array__text'


async def can_return_null(
    db_sesh: AsyncSession, 
) -> Union[str, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'can_return_null')()
        )
    )).scalar_one_or_none()
    return r

async def complex_id(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_id')(sqlalchemy.literal(None if z is None else z.model_dump(), type_=None))
        )
    )).scalar_one_or_none()
    return pydantic.TypeAdapter(Model__complex | None).validate_python(None if r is None else dict(r.items()))

async def count_leagues(
    db_sesh: AsyncSession, 
) -> Union[pydantic.JsonValue, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'count_leagues')()
        )
    )).scalar_one_or_none()
    return r

async def count_leagues_by_nullable(
    db_sesh: AsyncSession, _nullable: Union[str, None]
) -> Union[int, None]:
    'Count leagues by nullable'
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'count_leagues_by_nullable')(sqlalchemy.literal(_nullable, type_=postgresql.TEXT))
        )
    )).scalar_one_or_none()
    return r

async def do_anyrange(
    db_sesh: AsyncSession, r: Any
) -> Union[None, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'do_anyrange')(sqlalchemy.literal(r, type_=None))
        )
    )).scalar_one_or_none()
    return r

async def get_lists(
    db_sesh: AsyncSession, _list: Array__text
) -> Iterable[Array__text]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_lists')(sqlalchemy.literal(_list, type_=postgresql.ARRAY(postgresql.TEXT)))
        )
    )).scalars()
    return r

async def get_mood(
    db_sesh: AsyncSession, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_mood')(sqlalchemy.literal(_mood, type_=postgresql.ENUM(name='mood')))
        )
    )).scalar_one_or_none()
    return r

async def get_range(
    db_sesh: AsyncSession, 
) -> Any:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_range')()
        )
    )).scalar_one_or_none()
    return r

async def getall(
    db_sesh: AsyncSession, 
) -> Iterable[Model__league | None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'getall')()
        )
    )).scalars()
    return (pydantic.TypeAdapter(Model__league | None).validate_python(None if i is None else dict(i.items()))for i in r)

async def ids(
    db_sesh: AsyncSession, 
) -> Iterable[Union[int, None]]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'ids')()
        )
    )).scalars()
    return r

async def nullables(
    db_sesh: AsyncSession, 
) -> Iterable[Union[str, None]]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'nullables')()
        )
    )).scalars()
    return r

async def retvoid(
    db_sesh: AsyncSession, 
) -> Union[None, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'retvoid')()
        )
    )).scalar_one_or_none()
    return r

async def unitthing(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'unitthing')(sqlalchemy.literal(None if z is None else z.model_dump(), type_=None))
        )
    )).scalar_one_or_none()
    return pydantic.TypeAdapter(Model__complex | None).validate_python(None if r is None else dict(r.items()))
