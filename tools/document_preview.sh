#!/bin/bash

echo $1
pip install /paddle/build/opt/paddle/share/wheels/paddlepaddle-0.10.0-cp27-cp27mu-linux_x86_64.whl
apt-get update && apt-get install -y python-dev build-essential
cd /home/PaddlePaddle.org/portal
pip install -r requirements.txt
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle /home/FluidDoc &
echo 1111
