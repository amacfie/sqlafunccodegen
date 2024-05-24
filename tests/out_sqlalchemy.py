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


def can_return_null(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'can_return_null')()

def complex_id(
    z: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'complex_id')(z)

def count_leagues(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'count_leagues')()

def count_leagues_by_nullable(
    _nullable: Any
) -> Any:
    'Count leagues by nullable'
    return getattr(sqlalchemy.func, 'count_leagues_by_nullable')(_nullable)

def do_anyrange(
    r: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'do_anyrange')(r)

def get_lists(
    _list: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_lists')(_list)

def get_mood(
    _mood: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_mood')(_mood)

def get_range(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_range')()

def getall(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'getall')()

def ids(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'ids')()

def nullables(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'nullables')()

def retvoid(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'retvoid')()

def unitthing(
    z: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'unitthing')(z)
