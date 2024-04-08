# python simulation_CEVSV.py --M 1000 --L 2000 --N 100


#!/bin/bash

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script from the same directory
python "$DIR/simulation_CEVSV.py" --M 1000 --L 2000 --N 100
