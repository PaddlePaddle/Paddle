if [[ $1 == "pull"  ]]; then
	git pull
fi
cd build && rm -rf *
cmake  -DCMAKE_INSTALL_PREFIX=./tmp -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DWITH_TESTING=ON -DWITH_PROFILER=ON -DWITH_MKL=ON -DWITH_INFERENCE_API_TEST=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON ..
make -j
make install -j
make -j fluid_lib_dist
pip install --force-reinstall --user /home/chuanqiw/paddle/intelpaddle-shanghai/build/tmp/opt/paddle/share/wheels/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl 
