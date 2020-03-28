#!/bin/bash
python -c "import paddle.fluid;print(\"test command line flag ok now!\")"

echo "WITH_GPU:${WITH_GPU}"
echo "WITH_DISTRIBUTE:${WITH_DISTRIBUTE}"

if [[ "${WITH_GPU}"x == "ON"x && "${WITH_DISTRIBUTE}"x == "ON"x ]]; then
    python -m paddle.distributed.launch dummy.py 
fi
