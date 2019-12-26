#!/bin/bash
set -e

ver=`python -c "import platform;print(platform.python_version())"`
ver=${ver:0:1}

if [ $ver == '3' ]; then
   echo "PLSC only support py2 now."
   exit 0
fi

# install plsc
python -m pip install plsc

# test
python -m paddle.distributed.launch \
        --selected_gpus="0,1" \
        --log_dir="/tmp/plsc.log" do_plsc.py

# cleanup
rm -rf /tmp/plsc.log
python -m pip uninstall -y plsc
