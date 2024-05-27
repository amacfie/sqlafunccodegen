# sqlafunccodegen

Generate type-annotated Python functions that wrap PostgreSQL functions, using
SQLAlchemy.
Like [sqlacodegen](https://github.com/agronholm/sqlacodegen)
but for functions instead of tables.

Benefit over PostgREST: You can call functions and execute other SQL within
a transaction.

Usage:
```bash
sqlafunccodegen --help
```

Capabilities:
* "sqlalchemy" mode: functions directly wrap `sqlalchemy.func.<function_name>`
  * no types, just parameter names
* "python" mode: functions execute a `sqlalchemy.select`
  * many basic types
  * enums
  * arrays
  * some pseudotypes such as `anyarray`
  * Pydantic models for user-defined composite types
  * set-returning functions return iterables
  * constraints in domains are not checked but the underlying type is used
  * the Python types may be too restrictive or not restrictive enough, the
    correspondence isn't perfect. some types aren't recognized and the generic
    form in which sqlafunccodegen attempts to send them to the database may not
    work.
* both modes:
  * uses asyncpg
  * comments as docstrings
  * functions with overloads not supported
  * `IN`, `INOUT`, and `VARIADIC` params not supported
  * default values are not available

Generated code dependencies:
* asyncpg
* Pydantic 2
* SQLAlchemy 2

## Example

```sql
create table league (
    id serial primary key,
    description text
);

create function count_leagues_by_description(_description text) returns integer
as $$
    select count(*) from league where description = _description;
$$ language sql;

comment on function count_leagues_by_description is
    'Count leagues with a given description';
```

"python" mode:

```python
...

async def count_leagues_by_description(
    db_sesh: AsyncSession, _description: Union[str, None]
) -> Union[int, None]:
    "Count leagues with a given description"
    return (
        await db_sesh.execute(
            sqlalchemy.select(
                getattr(sqlalchemy.func, "count_leagues_by_description")(
                    sqlalchemy.bindparam(
                        key=None, value=_description, type_=sqlalchemy.Text
                    )
                )
            )
        )
    ).scalar_one_or_none()


```

"sqlalchemy" mode:

```python
...

def count_leagues_by_description(_description: Any) -> Any:
    "Count leagues with a given description"
    return getattr(sqlalchemy.func, "count_leagues_by_description")(_description)
```
