# R 语言支持
在 R 中使用 PaddlePaddle

## 环境安装
首先确保已安装Python，假设路径为`/opt/python3.8`

使用Python安装Paddle
``` bash
/opt/python3.8/bin/python3.8 -m pip install paddlepaddle # CPU
/opt/python3.8/bin/python3.8 -m pip install paddlepaddle-gpu # GPU
```

安装r运行paddle预测所需要的库
``` r
install.packages("reticulate") # 调用Paddle
```

## 在 R 中使用Paddle预测
首先在 R 中引入paddle预测环境

``` r
library(reticulate)
use_python("/opt/python3.8/bin/python")

paddle <- import("paddle.base.core")
```

创建一个AnalysisConfig，用于设置预测引擎的各选项

``` r
config <- paddle$AnalysisConfig("")
```

禁用feed和fetch OP，以使用 zero copy 预测
``` r
config$switch_use_feed_fetch_ops(FALSE)
config$switch_specify_input_names(TRUE)
```

设置模型路径有两种形式：
- `model` 目录中存在一个模型文件和多个参数文件
- `model` 目录中存在一个模型文件`__model__`和一个参数文件`__params__`

分别对应如下设置

``` r
config$set_model("model")
config$set_model("model/__model__", "model/__params__")
```

其他一些配置选项及说明如下
``` r
config$enable_profile() # 打开预测profile
config$enable_use_gpu(gpu_memory_mb, gpu_id) # 开启GPU预测
config$disable_gpu() # 禁用GPU
config$gpu_device_id() # 返回使用的GPU ID
config$switch_ir_optim(TRUE) # 开启IR优化(默认开启)
config$enable_tensorrt_engine(workspace_size,
                              max_batch_size,
                              min_subgraph_size,
                              paddle$AnalysisConfig$Precision$Float32,
                              use_static,
                              use_calib_mode
                              ) # 开启TensorRT
config$enable_mkldnn() # 开启MKLDNN
config$disable_glog_info() # 禁用预测中的glog日志
config$delete_pass(pass_name) # 预测的时候删除指定的pass

```

创建预测引擎
``` r
predictor <- paddle$create_paddle_predictor(config)
```

获取输入tensor(为简单起见，此处假设只有一个输入)，并设置输入tensor中的数据(注意需要使用np_array以传入numpy.ndarray类型的数据)
``` r
input_names <- predictor$get_input_names()
input_tensor <- predictor$get_input_tensor(input_names[1])

input_shape <- as.integer(c(1, 3, 300, 300)) # shape 为int类型
input_data <- np_array(data, dtype="float32")$reshape(input_shape)
input_tensor$copy_from_cpu(input_data)
```

运行预测引擎
``` r
predictor$zero_copy_run()
```

获取输出tensor(为简单起见，此处假设只有一个输出)
``` r
output_names <- predictor$get_output_names()
output_tensor <- predictor$get_output_tensor(output_names[1])
```

获取输出tensor中的数据，注意需要转为numpy.ndarray
``` r
output_data <- output_tensor$copy_to_cpu()
output_data <- np_array(output_data)
```

点击查看完整的[R预测示例](./example/mobilenet.r)及对应的[python预测示例](./example/mobilenet.py)

### 快速运行
将[Dockerfile](./Dockerfile)和[example](./example)下载到本地，使用以下命令构建docker镜像
``` bash
docker build -t paddle-rapi:latest .
```

启动一个容器
``` bash
docker run --rm -it paddle-rapi:latest bash
```

运行示例
``` bash
cd example && chmod +x mobilenet.r
./mobilenet.r
```
