
#!/bin/bash

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# # Run the Python script from the same directory

python simulation_CEVSV.py --M 2000 --L 6000 --N 200

# python "$DIR/simulation_CEVSV.py" --M 500 --L 1000 --N 50

# python "$DIR/simulation_CEVSV.py" --M 200 --L 400 --N 20

# python "$DIR/simulation_CEVSV.py" --M 100 --L 200 --N 10


