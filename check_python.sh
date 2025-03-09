#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Installing..."
    apt-get update && apt-get install -y python3
else
    echo "Python is installed."
fi

# Check if Python is in the PATH
if ! which python &> /dev/null; then
    echo "Python is not in the PATH. Adding..."
    # This should not be necessary if Python is installed correctly
    # However, you can add it to the PATH if needed
    export PATH=$PATH:/usr/bin/python
else
    echo "Python is in the PATH."
fi
