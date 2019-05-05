#!/bin/bash
cd /home
git clone https://github.com/PaddlePaddle/FluidDoc
git clone https://github.com/tianshuo78520a/PaddlePaddle.org.git
sh /home/FluidDoc/doc/fluid/api/gen_doc.sh
pip install /paddle/build/opt/paddle/share/wheels/*.whl
apt-get update && apt-get install -y python-dev build-essential
cd /home/PaddlePaddle.org/portal/portal
cd /home/PaddlePaddle.org/portal
pip install -r requirements.txt
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle /home/FluidDoc &
