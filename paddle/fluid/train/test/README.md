
### step 1. build paddle lib

```
cmake .. -DCMAKE_INSTALL_PREFIX=/paddle/src/dir/paddle/fluid/train/lib
make -j4 inference_lib_dist
```


### step 2. copy lib to this dir

```
cpy -r /paddle/src/dir/paddle/fluid/train/lib .
```

### step 3. generate program desc
```
python demo_network.py
```

This will generate two files:
  - startup_program: used to init all parameters
  - main_program: main logic of the network

### step 4. build demo_trainer and run it.

```
mkdir build
cd build
cmake ..
make
cp ../startup_program .
cp ../main_program .
./demo_trainer

```
