# Paddle神经网络拓扑表示方式重构

## 背景

目前Paddle中，解析用户配置的过程非常繁复。这也是因为Paddle作为一个四年左右项目的遗留问题。为了**兼容**之前所有的Paddle配置文件格式，也为了简化用户配置流程，现阶段Paddle共有三种配置风格。最原始的配置文件格式(`config_parser.py`)，`trainer_config_helper`和`paddle.v2.layer`。三者的调用关系为 `paddle.v2` 调用 `trainer_config_helper`再调用`config_parser.py`。虽然我们没有重复的写这些代码，但是多层的封装让代码很难维护。

主要痛点在于:

* 用户使用Layer，想去查询某一个参数应该如何使用。深入调研Paddle的代码会非常迷惑。同时，这些代码中只有`trainer_config_helper`是具有良好注释和良好风格的。`paddle.v2`虽然也有注释与文档，但其函数是动态生成的，而不是静态的代码，所以也不能**阅读**，而`config_parser.py`缺乏文档和注释。
* 开发者如果想要新写一个Layer，需要修改多个文件。开发者如果需要修改一个Layer的实现，也有同样的问题。
   * 首先，新写一个Layer，开发者需要在Paddle的[protobuf文件](https://github.com/PaddlePaddle/Paddle/blob/develop/proto/ModelConfig.proto)中，添加这个Layer需要的参数。
   * 其次，完成这个Layer需要的C/C++文件。完成这个Layer的前向后向代码。
   * 最后完成这个Layer配置文件解析`config_parser.py`，`trainer_config_helpers`和`paddle.v2`。
* 如果有其他Language Binding，需要开发的工作量太高。
* 使用protobuf作为多语言接口的中间协议，序列化速度慢，且对于嵌入式设备库大小过大。

所以这个设计的目标就是**治理**目前Paddle定义Layer和配置混乱复杂的问题，得到一个清爽的结果, **用户只需要**写一个`C/C++`实现即可完成一个Layer的开发。

同时，这个设计还会兼顾的问题有:

* 向后兼容性 ---- 即是否兼容之前的配置方式.
* 动态网络开发 ---- 神经网络配置解析为动态网络的基础部分。动态网络要求**配置解析必须快**。详细关于动态网络介绍，请参考[DynamicNet](./dynamic_net/00.how_to_implenment_dynamic_net.md)


## 实现方案

### 主要思路

* 在Paddle C++ Core中，开发新的数据结构表示神经网络的拓扑结构。第三方语言可以**直接**操纵C API来配置神经网络拓扑结构。
	* C++中新建一个namespace，叫做`paddle::topology`。
	* 新建的数据结构可以被序列化成`json`而不是`protobuf`，方便嵌入式部署。
* 开发一个新的存储元信息(`meta`)的数据结构，表示一个神经网络拓扑结构可以有哪些参数。
	* 这个元信息
	* 存储元信息的数据结构不要求可以被序列化和反序列化，因为元信息不影响模型的部署。但存储元信息的数据结构可以被C API暴露给第三方语言(例如Python)。
	* 第三方语言读取拓扑结构元信息，进而在第三方语言中生成配置拓扑结构的函数。
* 目前Paddle中的`protobuf`格式，可以转换为新的拓扑结构。进而维持原有代码的向后兼容性。让Paddle可以渐进式优化。

### 具体实现

#### 元信息

所有的元信息类放置到`paddle::topology::meta`名字空间下。基本的元信息包括如下几个方面:

##### AttributeMeta

`AttributeMeta`表示拓扑结构中所有属性的元信息。属性是指拓扑结构中配置的参数。例如层的大小，激活函数的形式等等。这些属性的元信息都由`AttributeMeta`表示。

```cpp
class AttributeMeta {
public:
  std::string name;  // attribute name, e.g., 'size' in layer.
  std::type_info type;  // attribute type, e.g., 'uint64_t' about 'size'
  std::string description; // the description of this attribute, e.g., 'the size of layer'.
  std::any checkCallback; // The function check whether this attribute is valid or not.
};
```
其中 checkCallback是一个回调函数，他的类型是(T* attr, bool setted) => paddle::Error，因为输入的attr可以是任意类型的泛型，故这里用std::any表示类型。其中setted是表示这个参数是不是被用户设置过。

举例对于layer的`dropout_rate`这个属性的AttributeMeta可以设置为:

```cpp
auto dropoutMeta = new AttributeMeta();
dropoutMeta->name = "dropout";
dropoutMeta->type = typeid(float);
dropoutMeta->description = "Set drop out rate of layer. "
                            "1 means all activations are dropped, 0 means do not drop any activation";
dropoutMeta->checkCallback = [](float* attr, bool setted) -> paddle::Error {
   if (!setted) {
     *attr = 0.0f;  // default value;
     return paddle::Error::OK;
   }
   // Check whether input is valid.
   if (*attr < 0.0f || *attr > 1.0f) {
   	 return paddle::Error("dropout rate should be in [0.0, 1.0]");
   } else {
     return paddle::Error::OK;
   }
};
```

##### TensorMeta

`TensorMeta`表示一个神经网络拓扑结构中，每一层的输入信息和参数信息的元信息(对于于目前Paddle C++ Core中的Parameter和Argument的元信息)。

`TensorMeta`由许多`AttributeMeta`构成。

```cpp
class TensorMeta {
public:
  std::vector<AttributeMeta> attributes;
};
```

举例说明，对于全连接层的输入，可能的`TensorMeta`值为:

```cpp
enum DataType {
  DENSE=0,
  SPARSE_BINARY,
  SPARSE,
  INTEGER
};

enum SequenceType {
  NO_SEQUENCE=0,
  SEQUENCE,
  NESTED_SEQUENCE
};
...

auto inputMeta = new TensorMeta();
auto dataTypeMeta = new AttrbuteMeta("type", typeid(std::pair<DataType, SequenceType>), "Data type of this tensor");
dataTypeMeta->checkCallback = [](std::pair<DataType, SequenceType>* type, bool setted) -> paddle::Error {
  if (!setted) {
    return paddle::Error("Type of tensor should be setted");
  }
  if (*type != {DENSE, NO_SEQUENCE}) {
    return paddle::Error("FC Layer only support dense, no_sequence data type as input.");
  }
  return paddle::Error::OK;
}
inputMeta->attributes.push_back(dataTypeMeta);

auto shapeMeta = new AttributeMeta("shape", typeid(std::vector<uint32_t>), "The shape of this tensor");
shapeMeta->checkCallback = [](std::vector<uint32_t>* shape, bool setted) {
  if (!setted) {
    return paddle::Error("Shape of tensor should be setted");
  }
  if (shape->size() != 2) {
  	return paddle::Error("FC Layer only support 2 dim tensor(a.k.a matrix) as input.");
  }
  if (shape->at(1) > 0) {
  	return paddle::Error("The width of fc layer input should larger than 0.");
  }
  return paddle::Error::OK;
}
inputMeta->attributes.push_back(inputMeta);
```

对于全连接层的输入参数，可能的`TensorMeta`值为:

```cpp
auto inputParamMeta = new TensorMeta();
inputMeta->attributes.push_back(new AttributeMeta("shape", ...));
inputMeta->attributes.push_back(new AttributeMeta("type", ..));
inputMeta->attributes.push_back(new AttributeMeta("weight_decay", ...)); // support weight decay;
inputMeta->attributes.push_back(new AttributeMeta("initial_mean", ...)); // init strategy;
inputMeta->attributes.push_back(new AttributeMeta("initial_std", ...)); // init std.
...
```

##### LayerInputMeta

神经网络层输入的元信息。一个层的输入既包括了某一个层输入的值，也包括了和这个输入配合的参数值。某些层的输入可以没有参数。某些层的输入可以是无穷多个，但是每一个输入都使用同一个`LayerInputMeta`.

```cpp

class LayerInputMeta {
public:
  std::string name;
  std::string description;
  bool canBeMany; // some layer can have unlimited number of input, but share same meta.
  TensorMeta inputTensorMeta;
  std::unique_ptr<TensorMeta> paramTensorMeta;  // could be null;
}
```

举例FC Layer的`LayerInputMeta`为:

```cpp
auto fcInputMeta = new LayerInputMeta();
fcInputMeta->name = "input";
fcInputMeta->description = "The input of fully connected layer";
fcInputMeta->canBeMany = true;
fcInputMeta->inputTensorMeta = fcInputTensorMeta;
fcInputMeta->inputParameterMeta = fcInputParamMeta;
```

##### LayerMeta

LayerMeta表示一个神经网络层可以的元信息。它包括这个层的类型，描述，这个层输入的元信息，bias参数的元信息和这个层的一些其他属性(例如dropout_rate)。

```cpp
struct LayerMeta {
	std::string type;
	std::string description;
	std::vector<LayerInputMeta> inputs;
	std::unique_ptr<TensorMeta> bias;
	std::vector<AttributeDef> attrs;
};
```

实际举例略，和`LayerInputMeta`等类似。

##### TopologyMeta

TopologyMeta表示一个拓扑结构可以设置的属性。他包括Paddle支持的层的信息，也包括拓扑结构中可以配置的其他属性。例如输入层的名字等等。

```cpp
struct TopologyMeta {
  std::vector<LayerMeta> layers;
  std::vector<AttributeMeta> attrs;
};

```

#### 实际信息

实际信息指每一个神经网络拓扑结构的实际描述，也表示着神经网络拓扑结构中的实际参数。它是可以被序列化成`json`的数据结构。也是每次训练或者预测时，Paddle载入的实际模型配置。这些信息被放置在`paddle::topology`名字空间下。

##### Attribute

神经网络的属性可以是任意类型，使用`std::pair<std::string, std::any>`表示。不再创建新的类型。


##### Tensor

Tensorl表示拓扑结构中某一个Tensor实际配置属性。

```cpp
class Tensor {
public:
  std::unordered_map<std::string, std::any> attributes;
};
```

##### LayerInput

`LayerInput`表示神经网络某一层的实际输入配置。

```cpp
class LayerInput {
public:
  std::string name;
  Tensor inputTensor;
  std::unique_ptr<Tensor> paramTensor;  // could be null;
}
```

##### Layer

`Layer`表示神经网络某一个层的实际配置。

```cpp
class Layer {
public:
	std::vector<LayerInput> inputs;
	std::unique_ptr<Tensor> bias;
	std::unordered_map<std::string, std::any> attrs;
};

```

##### Topology

`Topology`表示一个神经网络Topology的全部配置。即为Paddle中的`Topology`类。

```cpp
class Topology {
public:
  std::vector<Layer> layers;
  std::unordered_map<std::string, std::any> attrs;
}
```
