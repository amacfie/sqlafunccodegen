#!/usr/bin/env bash

docker run -d --name testdb --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-bookworm
sleep 5
psql postgresql://postgres:postgres@localhost:5432/postgres -f tests/schema.ddl

python -m unittest
TESTS_EXIT_CODE=$?

docker kill testdb
exit $TESTS_EXIT_CODE
