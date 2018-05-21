
### step 1. build paddle lib

```
PADDLE_LIB=/paddle/lib/dir

cmake .. -DCMAKE_INSTALL_PREFIX=$PADDLE_LIB \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_FLUID_ONLY=ON
         -DWITH_MKL=OFF \ # or ON
         -DWITH_GPU=OFF
         -DWITH_STYLE_CHECK=OFF

make -j4 inference_lib_dist
```

### step 2. generate program desc
```
python demo_network.py
```

This will generate two program desc files:
  - startup_program: used to init all parameters
  - main_program: main logic of the network

### step 3. build demo_trainer and run it.

```
mkdir build
cd build

PADDLE_LIB=/paddle/lib/dir

# PADDLE_LIB is the same with CMAKE_INSTALL_PREFIX when building the lib
cmake .. -DPADDLE_LIB=$PADDLE_LIB \
         -DWITH_MKLDNN=OFF \ # or ON
         -DWITH_MKL=OFF \ # or ON
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
