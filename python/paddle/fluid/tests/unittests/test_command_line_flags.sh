#!/bin/bash
python -c "import paddle.fluid;print(\"test command line flag ok now!\")"
python -m paddle.distributed.launch dummy.py 
