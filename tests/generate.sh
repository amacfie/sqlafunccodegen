#!/usr/bin/env bash

docker run -d --name testdb --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-bookworm
sleep 5
psql postgresql://postgres:postgres@localhost:5432/postgres -f tests/schema.ddl
cp tests/out_python.py tests/out_python.py.bak
cp tests/out_func.py tests/out_func.py.bak
cp tests/out_asyncpg_only.py tests/out_asyncpg_only.py.bak
python sqlafunccodegen/main.py --mode python > tests/out_python.py
python sqlafunccodegen/main.py --mode func > tests/out_func.py
python sqlafunccodegen/main.py --mode asyncpg_only > tests/out_asyncpg_only.py
docker kill testdb
diff tests/out_python.py tests/out_python.py.bak
diff tests/out_func.py tests/out_func.py.bak
diff tests/out_asyncpg_only.py tests/out_asyncpg_only.py.bak
