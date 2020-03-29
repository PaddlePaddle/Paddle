#!/bin/bash
python -c "import paddle.fluid;print(\"test command line flag ok now!\")"

echo "WITH_GPU:${WITH_GPU}"
echo "WITH_DISTRIBUTE:${WITH_DISTRIBUTE}"

echo "import paddle.fluid as fluid" > ../paddle_mod_dummy.py
python -m paddle.fluid.tests.paddle_mod_dummy
