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
```

->

```python
...

async def count_leagues_by_description(
    db_sesh: AsyncSession, _description: Union[str, None]
) -> Union[int, None]:
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