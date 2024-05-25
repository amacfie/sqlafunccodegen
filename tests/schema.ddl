CREATE TYPE complex AS (
    r       float,
    i       float
);
comment on type complex is 'A complex number';
comment on column complex.r is 'The real part';

create table league (
    id serial primary key,
    name text not null,
    nullable text,
    stuff text[],
    cs complex[]
);

create domain unit_complex as complex check (
  sqrt((VALUE).r*(VALUE).r + (VALUE).i*(VALUE).i) = 1
);

CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');

create type c2vector as (
    z1 complex,
    z2 complex,
    moods mood[]
);


insert into league (name, nullable) values('Premier League', null);
insert into league (name, nullable, cs) values(
    'Bundesliga', 'extra', array[(10, 20), (30, 40)]::complex[]);


create function c2vector_id(c c2vector) returns c2vector as $$
    select c;
$$ language sql;

create function all_leagues() returns setof league as $$
    select * from league order by id;
$$ language sql;

create function set_of_complex_arrays() returns setof complex[] as $$
    select * from (values
        (array[(1.0, 2.0), (3.0, 4.0)]::complex[]),
        (array[(5.0, 6.0), (7.0, 8.0)]::complex[])
    ) as t;
$$ language sql;

create function complex_array_id(ca complex[]) returns complex[] as $$
    select ca;
$$ language sql;

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

create function get_stuff(_stuff text[]) returns setof text[] as $$
    select stuff from league where stuff && _stuff;
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

create function complex_id(z complex) returns complex as $$
    select z;
$$ language sql;

create function unitthing(z unit_complex) returns unit_complex as $$
    select z;
$$ language sql;

create function array_id(arr int[]) returns int[] as $$
    select arr;
$$ language sql;

create function first_any(a anyelement, b anyarray) returns anyelement as $$
    select a;
$$ language sql;

create function circle_id(c circle) returns circle as $$
    select c;
$$ language sql;

create function anyenum_f(a anyenum, b anyarray) returns anyenum as $$
    select a;
$$ language sql;

create function jsonb_id(j jsonb) returns jsonb as $$
    select j;
$$ language sql;


