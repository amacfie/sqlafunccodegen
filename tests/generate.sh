#!/usr/bin/env bash

set -e

docker run -d --name testdb --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-bookworm
sleep 5
psql postgresql://postgres:postgres@localhost:5432/postgres -f tests/schema.ddl
cp tests/out_python.py tests/out_python.py.bak
cp tests/out_sqlalchemy.py tests/out_sqlalchemy.py.bak
python sqlafunccodegen/main.py --mode python > tests/out_python.py
python sqlafunccodegen/main.py --mode sqlalchemy > tests/out_sqlalchemy.py
docker kill testdb
diff tests/out_python.py tests/out_python.py.bak
diff tests/out_sqlalchemy.py tests/out_sqlalchemy.py.bak
