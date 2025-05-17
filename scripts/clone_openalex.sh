#!/bin/bash

set -e
# Based on https://docs.openalex.org/download-all-data/upload-to-your-database/load-to-a-relational-database
source .env

# Variables
SQL_URL="https://raw.githubusercontent.com/ourresearch/openalex-documentation-scripts/main/openalex-pg-schema.sql"
SQL_FILE="tmp/openalex-pg-schema.sql"

FLATTEN_PY_URL="https://raw.githubusercontent.com/ourresearch/openalex-documentation-scripts/refs/heads/main/flatten-openalex-jsonl.py"
FLATTEN_PY_FILE="tmp/flatten-openalex-jsonl.py"

COPY_SQL_URL="https://raw.githubusercontent.com/ourresearch/openalex-documentation-scripts/refs/heads/main/copy-openalex-csv.sql"
COPY_SQL_FILE="tmp/copy-openalex-csv.sql"


# Download SQL file
wget -O $SQL_FILE $SQL_URL
wget -O $FLATTEN_PY_FILE $FLATTEN_PY_URL
wget -O $COPY_SQL_FILE $COPY_SQL_URL

# Step 1: Create the schema
psql $POSTGRES_URL -f $SQL_FILE

# Download compressed data form S3 bucket
mc alias set openalex https://s3.amazonaws.com "" ""
mc mirror --overwrite --remove openalex/openalex openalex-snapshot

# Step 2: Convert the JSON Lines files to CSV
python $FLATTEN_PY_FILE

# Step 3: Load the CSV files into the database
psql $POSTGRES_URL < $COPY_SQL_FILE