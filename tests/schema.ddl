-- docker run --name testdb --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-bookworm
-- psql postgresql://postgres:postgres@localhost:5432/postgres -f tests/schema.ddl

create table league (
    id serial primary key,
    name text not null,
    nullable text,
    list text[]
);

insert into league (name, nullable) values('Premier League', null);
insert into league (name, nullable) values('Bundesliga', 'extra');

create function count_leagues() returns json as $$
    select '{"count": 4}'::json;
$$ language sql;

create function can_return_null() returns text as $$
    select nullable from league limit 1;
$$ language sql;

create function count_leagues_by_nullable(_nullable text) returns integer as $$
    select count(*) from league where nullable = _nullable;
$$ language sql;

comment on function count_leagues_by_nullable(text) is 'Count leagues by nullable';

create function ids() returns setof integer as $$
    select id from league;
$$ language sql;

create function overloaded(x int) returns void as $$
    select null;
$$ language sql;
create function overloaded(x text) returns void as $$
    select null;
$$ language sql;

create function nullables() returns setof text as $$
    select nullable from league;
$$ language sql;

create function getall() returns setof league as $$
    select * from league;
$$ language sql;

create function get_lists(_list text[]) returns setof text[] as $$
    select list from league where list && _list;
$$ language sql;

create function retvoid() returns void as $$
    update league set name = name;
$$ language sql;

CREATE OR REPLACE FUNCTION sum_variadic(VARIADIC numbers int[])
RETURNS int AS $$
BEGIN
    RETURN (SELECT SUM(x) FROM unnest(numbers) AS x);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_test(OUT x text, OUT y text)
AS $$
BEGIN
   x := 1;
   y := 2;
END;
$$  LANGUAGE plpgsql;

create or replace function swap(
   a int,
	inout x int,
	inout y int,
	b int
)
language plpgsql
as $$
begin
   select x,y into y,x;
end; $$;


CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');

CREATE OR REPLACE FUNCTION get_mood(_mood mood) RETURNS mood AS $$
BEGIN
    RETURN 'happy';
END;
$$ LANGUAGE plpgsql;

create function get_range() returns int4range as $$
      select int4range(1, 10);
$$ language sql;

create function do_anyrange(r anyrange) returns void as $$
      select null;
$$ language sql;

CREATE TYPE complex AS (
    r       float,
    i       float
);

create function complex_id(z complex) returns complex as $$
    select z;
$$ language sql;

-- domain of complex numbers on unit circle
create domain unit_complex as complex check (
  sqrt((VALUE).r*(VALUE).r + (VALUE).i*(VALUE).i) = 1
);

create function unitthing(z unit_complex) returns unit_complex as $$
    select z;
$$ language sql;
