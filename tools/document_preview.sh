#!/bin/bash
PADDLE_ROOT=/home
mkdir ${PADDLE_ROOT}
cd ${PADDLE_ROOT}
pip install /paddle/build/opt/paddle/share/wheels/*.whl
git clone https://github.com/PaddlePaddle/FluidDoc
git clone https://github.com/tianshuo78520a/PaddlePaddle.org.git
cd ${PADDLE_ROOT}/FluidDoc
export GIT_SSL_NO_VERIFY=1
git pull
cd ${PADDLE_ROOT}/FluidDoc/doc/fluid/api
sh gen_doc.sh
cd  ${PADDLE_ROOT}/PaddlePaddle.org
git reset 3feaa68376d8423e41d076814e901e6bf108c705
cd ${PADDLE_ROOT}/PaddlePaddle.org/portal
pip install -r requirements.txt
#If the default port is not occupied, you can use port 8000, you need to replace it with a random port on the CI.
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle ${PADDLE_ROOT}/FluidDoc &
