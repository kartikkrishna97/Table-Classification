#!/bin/bash
echo "$2"
if [[ $# -eq 0 || $1 != "test" ]]; then
    echo "Training the model"
    python train_column.py "$1"
    python infer.py test.pth "$2"

elif [ "$1" == "test" ]; then
    python test.py test.pth "$3" "$4" 

else
    echo "Invalid mode"

fi
