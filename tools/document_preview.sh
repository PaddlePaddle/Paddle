#!/bin/bash
cd /home
git clone https://github.com/PaddlePaddle/FluidDoc
git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git
sh /home/FluidDoc/doc/fluid/api/gen_doc.sh
pip install /paddle/build/opt/paddle/share/wheels/*.whl
apt-get update && apt-get install -y python-dev build-essential
cd /home/PaddlePaddle.org/portal/portal
sed -i "210a return 'http://paddlepaddle.org'" documentation_generator.py
sed -i "211s#^#        #g" documentation_generator.py
cd /home/PaddlePaddle.org/portal
pip install -r requirements.txt
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle /home/FluidDoc &
