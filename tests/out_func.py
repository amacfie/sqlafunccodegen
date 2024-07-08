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
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
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
def all_leagues(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'all_leagues')()

def array_id(
    arr: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'array_id')(arr)

def c2vector_id(
    c: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'c2vector_id')(c)

def can_return_null(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'can_return_null')()

def circle_id(
    c: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'circle_id')(c)

def complex_array_id(
    ca: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'complex_array_id')(ca)

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

def get_mood(
    _mood: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_mood')(_mood)

def get_range(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_range')()

def get_stuff(
    _stuff: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_stuff')(_stuff)

def getall(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'getall')()

def ids(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'ids')()

def jsonb_id(
    j: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'jsonb_id')(j)

def nullables(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'nullables')()

def retvoid(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'retvoid')()

def set_of_complex_arrays(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'set_of_complex_arrays')()

def unitthing(
    z: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'unitthing')(z)
