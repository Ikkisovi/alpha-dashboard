#!/bin/bash
set -e

# Define source roots
REPO_ROOT=".."
DEST_ROOT="."

echo "Cleaning up previous staging..."
rm -rf "$DEST_ROOT/src"
rm -rf "$DEST_ROOT/python-backend"
rm -rf "$DEST_ROOT/data"

echo "Creating directories..."
mkdir -p "$DEST_ROOT/src"
mkdir -p "$DEST_ROOT/python-backend"
mkdir -p "$DEST_ROOT/data"

echo "Copying src..."
cp -r "$REPO_ROOT/src/"* "$DEST_ROOT/src/"

echo "Copying python-backend..."
cp -r "$REPO_ROOT/python-backend/"* "$DEST_ROOT/python-backend/"

echo "Copying Qlib data (1555_qlib only)..."
cp -r "$REPO_ROOT/data/1555_qlib" "$DEST_ROOT/data/"

echo "Copying top_5_factors_20d.json..."
cp "$REPO_ROOT/data/top_5_factors_20d.json" "$DEST_ROOT/data/"

echo "Selecting best pool_100000.json..."
# Logic to copy the largest pool file as pool_100000.json
# We use 'ls -S' to sort by size, head -1 to get largest
BEST_POOL=$(find "$REPO_ROOT/data" -name "pool_100000.json" -type f -exec ls -S {} + | head -1)
if [ -n "$BEST_POOL" ]; then
    echo "Found best pool: $BEST_POOL"
    cp "$BEST_POOL" "$DEST_ROOT/data/pool_100000.json"
else
    echo "Warning: No pool_100000.json found."
fi

echo "Preparation complete. Asset bundle ready in $DEST_ROOT."
