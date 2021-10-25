set -e
output=$(python3 ./envs.py $*)
eval $output
