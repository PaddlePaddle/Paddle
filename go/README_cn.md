# Paddle 预测golang API

## 安装
首先cmake编译时打开`-DON_INFER=ON`,在编译目录下得到``fluid_inference_c_install_dir``,将该目录移动到当前目录中并重命名为`paddle_c`

## 在Go中使用Paddle预测
首先创建预测配置
``` go
config := paddle.NewAnalysisConfig()
config.SetModel(model_file, params_file)
config.SwitchUseFeedFetchOps(false)
config.SwitchSpecifyInputNames(true)
```

创建predictor
``` go
predictor := paddle.NewPredictor(config)
```

获取输入Tensor和输出Tensor
``` go
inputs = predictor.GetInputTensors()
```

设置输入数据(假设只有一个输入)
``` go
input := inputs[0]
input.SetValue(data)
input.Reshape([]int32{1, 3, 300, 300})
```

运行预测
``` go
predictor.ZeroCopyRun()
```

获取输入Tensor的真实值
``` go
output := outputs[0]
predictor.GetZeroCopyOutput(output)
value := reflect.ValueOf(output.Value())
shape, dtype := paddle.ShapeAndTypeOf(value)
output_data := value.Interface().([][]float32)
```

## 示例
源码见[mobilenet](./demo/mobilenet.go)

下载[数据](https://paddle-inference-dist.cdn.bcebos.com/mobilenet-test-model-data.tar.gz)并解压到当前目录

运行
``` go
go run ./demo/mobilenet.go
```
