#!/bin/bash

# Read the file path from the .txt file and assign it to a variable
file_path=$(<path.txt)

# Add the file path to the PYTHONPATH environment variable
export PYTHONPATH="$file_path:$PYTHONPATH"

# Optionally, print the updated PYTHONPATH to verify
echo "Updated PYTHONPATH: $PYTHONPATH"

