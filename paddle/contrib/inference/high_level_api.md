# Inference High-level APIs
This document describes the high-level inference APIs one can use to easily deploy a Paddle model for an application.

The APIs are described in `paddle_inference_api.h`, just one header file, and two libaries `libpaddle_fluid.so` and `libpaddle_fluid_api.so` are needed.

## PaddleTensor
We provide the `PaddleTensor` data structure is to give a general tensor interface.

The definition is 

```c++
struct PaddleTensor {
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
};
```

The data is stored in a continuous memory `PaddleBuf`, and tensor's data type is specified by a `PaddleDType`. 
The `name` field is used to specify the name of input variable, 
that is important when there are multiple inputs and need to distiuish which variable to set.

## engine
The inference APIs has two different underlying implementation, currently there are two valid engines:

- the native engine, which is consists of the native operators and framework,
- the Anakin engine, which is a Anakin library embeded.

The native engine takes a native Paddle model as input, and supports any model that trained by Paddle, 
but the Anakin engine can only take the Anakin model as input(user need to manully transform the format first) and currently not all Paddle models are supported.

```c++
enum class PaddleEngineKind {
  kNative = 0,  // Use the native Fluid facility.
  kAnakin,      // Use Anakin for inference.
};
```

## PaddlePredictor and how to create one
The main interface is `PaddlePredictor`, there are following methods 

- `bool Run(const std::vector<PaddleTensor>& inputs, std::vector<PaddleTensor>* output_data)`
  - take inputs and output `output_data`
- `Clone` to clone a predictor from an existing one, with model parameter shared.

There is a factory method to help create a predictor, and the user takes the ownership of this object.

```c++
template <typename ConfigT, PaddleEngineKind engine = PaddleEngineKind::kNative>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);
```

By specifying the engine kind and config, one can get an specific implementation.

## Reference

- [paddle_inference_api.h](./paddle_inference_api.h)
- [demos](./demo)
