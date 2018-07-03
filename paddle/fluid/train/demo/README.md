
### step 1. build paddle lib

```

# WITH_MKL=ON|OFF
# WITH_MKLDNN=ON|OFF

PADDLE_LIB=/paddle/lib/dir
cmake .. -DFLUID_INSTALL_DIR=$PADDLE_LIB \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_FLUID_ONLY=ON \
         -DWITH_GPU=OFF \
         -DWITH_STYLE_CHECK=OFF \
         -DWITH_MKL=OFF \
         -DWITH_MKLDNN=OFF
make -j8
make -j8 inference_lib_dist
```

### step 2. generate program desc
```
# please install paddle before run this scripe
pip install --upgrade paddlepaddle-*.whl
python demo_network.py
```

This will generate two program desc files:
  - startup_program: used to init all parameters
  - main_program: main logic of the network

### step 3. build demo_trainer and run it.


```
# Make a build dir at the same dir of this README.md document.
# The demo dir can be put anywhere.
mkdir build
cd build

# WITH_MKL=ON|OFF
# WITH_MKLDNN=ON|OFF
PADDLE_LIB=/paddle/lib/dir

# PADDLE_LIB is the same with FLUID_INSTALL_DIR when building the lib
cmake .. -DPADDLE_LIB=$PADDLE_LIB \
         -DWITH_MKLDNN=OFF \
         -DWITH_MKL=OFF
make

# copy startup_program and main_program to this dir
cp ../startup_program .
cp ../main_program .

# run demo cpp trainer
./demo_trainer

```

The output will be:
```
step: 0 loss: 1069.02
step: 1 loss: 1069.02
step: 2 loss: 1069.02
....
```
