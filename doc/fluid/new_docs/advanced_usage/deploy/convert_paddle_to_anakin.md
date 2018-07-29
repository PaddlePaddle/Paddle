# 模型转换指南

Anakin 支持不同框架的模型预测。但由于格式的差别，Anakin 需要您预先转换模型。本文档介绍如何转换模型。

## 简介

Anakin 模型转换器输入支持 Caffe 和 Fluid 两种格式的预测模型，模型包含网络结构（model 或 prototxt）和权重参数（param 或 caffemodel）。   

模型转换的输出是一个 bin 文件，它作为 Anakin 框架的 graph 参数导入。   

您还可以使用模型转换器的 launch board 功能生成网络结构的 HTML 预览。   


## 系统要求

- python 2.7+
- pyyaml
- flask
- protobuf 3.5+


## 用法

### 1、环境
转换器所需的依赖标注于 *系统要求* 一节。

### 2、配置
您需要对 *config.yaml* 文件进行修改以告知您的需求。工程中给出了 *config.yaml* 示例，下面作进一步说明。

#### config.yaml
```bash
OPTIONS:
    Framework: CAFFE       # 依框架类型填写 CAFFE 或 FLUID
    SavePath: ./output     # 转换结束后模型的保存位置
    ResultName: googlenet  # 输出模型的名字
    Config:
        LaunchBoard: ON    # 是否生成网络结构预览页面
        Server:
            ip: 0.0.0.0
            port: 8888     # 从一个可用端口访问预览页面
        OptimizedGraph:    # 当您使用了 Anakin 框架的 Optimized 功能时，才应该打开此项
            enable: OFF
            path: /path/to/anakin_optimized_anakin_model/googlenet.anakin.bin.saved
    LOGGER:
        LogToPath: ./log/  # 生成日志的路径
        WithColor: ON

TARGET:
    CAFFE:
        # 当 Framework 为 CAFFE 时需填写
        ProtoPaths:
            - /path/to/caffe/src/caffe/proto/caffe.proto
        PrototxtPath: /path/to/your/googlenet.prototxt
        ModelPath: /path/to/your/googlenet.caffemodel

    FLUID:
        # 当 Framework 为 FLUID 时需填写
        Debug: NULL
        ProtoPaths:
            - /
        PrototxtPath: /path/to/fluid/inference_model
        ModelPath: /path/to/fluid/inference_model
	# ...
```

### 3、转换
在完成配置文件的修改后，您只需执行 ```python converter.py``` 就可以进行模型转换了。


### 4、预览
最后一步，就是在浏览器中查看令人振奋的转换结果！网址是在 *config.yaml* 中配置的，例如 http://0.0.0.0:8888 。

> 注意：若您使用了默认的 IP 地址 0.0.0.0，请在预览时使用真实的服务器地址 real_ip:port 替代它。
