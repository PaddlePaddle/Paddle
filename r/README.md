# R support

English | [简体中文](./README_cn.md)

Use paddle in R.

## Install
### Use docker
Download [`Dockerfile`](./Dockerfile), run
``` bash
docker build -t paddle-rapi:latest .
```

### Local installation
First, make sure `Python` is installed, assuming that the path is `/opt/python3.7`.

``` bash
python -m pip install paddlepaddle # CPU version
python -m pip install paddlepaddle-gpu # GPU version
```

Install the R libraries needed to use paddle.
``` r
install.packages("reticulate") # call Python in R
install.packages("RcppCNPy") # use numpy.ndarray in R
```

## Use Paddle inference in R
First, load PaddlePaddle in R.
``` r
library(reticulate)
library(RcppCNPy)

use_python("/opt/python3.7/bin/python3.7")
paddle <- import("paddle.fluid.core")
```

Create an `AnalysisConfig`, which is the configuration of the paddle inference engine.
``` r
config <- paddle$AnalysisConfig("")
```

Set model path.
``` r
config$set_model("model/__model__", "model/__params__")
```

Use zero copy inference.
``` r
config$switch_use_feed_fetch_ops(FALSE)
config$switch_specify_input_names(TRUE)
```

Other configuration options and descriptions are as fallows.
``` r
config$enable_profile() # turn on inference profile
config$enable_use_gpu(gpu_memory_mb, gpu_id) # use GPU
config$disable_gpu() # disable GPU
config$gpu_device_id() # get GPU id
config$switch_ir_optim(TRUE) # turn on IR optimize(default is TRUE)
config$enable_tensorrt_engine(workspace_size,
                              max_batch_size,
                              min_subgraph_size,
                              paddle$AnalysisConfig$Precision$FLOAT32,
                              use_static,
                              use_calib_mode
                              ) # use TensorRT
config$enable_mkldnn() # use MKLDNN
config$delete_pass(pass_name) # delete IR pass
```

Create inference engine.
``` r
predictor <- paddle$create_paddle_predictor(config)
```

Get input tensor(assume single input), and set input data
``` r
input_names <- predictor$get_input_names()
input_tensor <- predictor$get_input_tensor(input_names[1])
input_shape <- as.integer(c(1, 3, 300, 300)) # shape has integer type
input_data <- np_array(data, dtype="float32")$reshape(input_shape)
input_tensor$copy_from_cpu(input_data)
```

Run inference.
``` r
predictor$zero_copy_run()
```

Get output tensor(assume single output).
``` r
output_names <- predictor$get_output_names()
output_tensor <- predictor$get_output_tensor(output_names[1])
```

Parse output data, and convert to `numpy.ndarray`
``` r
output_data <- output_tensor$copy_to_cpu()
output_data <- np_array(output_data)
```

Click to see the full [R mobilenet example](./example/mobilenet.r) and the corresponding [Python mobilenet example](./example/mobilenet.py) the above. For more examples, see [R inference example](./example).

## Quick start
Download [Dockerfile](./Dockerfile) and [example](./example) to local directory, and build docker image
``` bash
docker build -t paddle-rapi:latest .
```

Create and enter container
``` bash
docker run --rm -it paddle-rapi:latest bash
```

Run the following command in th container
```
cd example
chmod +x mobilenet.r
./mobilenet.r
```
