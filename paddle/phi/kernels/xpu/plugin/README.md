# XPU PLUGIN
## Standalone build and test.
```
$ cd plugin
Modify ./build.sh to set the path of XDNN, XRE and XTDK.
$ ./build.sh

$ cd example
Modify ./example/build.sh to set the path of XDNN and XRE.
$ ./build.sh
$ ./run.sh
```
## Build with PaddlePaddle.
### Copy to the source code of PaddlePaddle.
```
$ cp -rf plugin <path_to_paddle_source_code>/paddle/phi/xpu
```
### Add -DWITH_XPU_PLUGIN=ON as extra cmake arguments.
```
$ cmake .. <other_cmake_args> -DWITH_XPU_PLUGIN=ON
```
