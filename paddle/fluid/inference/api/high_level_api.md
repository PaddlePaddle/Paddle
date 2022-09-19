# Inference High-level APIs
This document describes the high-level inference APIs, one can use them to deploy a Paddle model for an application quickly.

The APIs are described in `paddle_inference_api.h`, just one header file, and two libaries `libpaddle_inference.so` and `libpaddle_inference_io.so` are needed for a deployment.

## PaddleTensor
We provide the `PaddleTensor` data structure to give a general tensor interface.

The definition is

```c++
struct PaddleTensor {
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
};
```

The data is stored in a continuous memory `PaddleBuf,` and a `PaddleDType` specifies tensor's data type.
The `name` field is used to specify the name of an input variable,
that is important when there are multiple inputs and need to distinguish which variable to set.

## engine
The inference APIs has two different underlying engines

- the native engine
- the tensorrt engine

The native engine, which is consists of the native operators and framework, takes a native Paddle model
as input, and supports any model that trained by Paddle.

```c++
enum class PaddleEngineKind {
  kNative = 0,  // Use the native Fluid facility.
  kAutoMixedTensorRT // Automatically mixing TensorRT with the Fluid ops.
};
```

## PaddlePredictor and how to create one
The main interface is `PaddlePredictor,` there are following methods

- `bool Run(const std::vector<PaddleTensor>& inputs, std::vector<PaddleTensor>* output_data)`
  - take inputs and output `output_data.`
- `Clone` to clone a predictor from an existing one, with model parameter shared.

There is a factory method to help create a predictor, and the user takes the ownership of this object.

```c++
template <typename ConfigT, PaddleEngineKind engine = PaddleEngineKind::kNative>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);
```

By specifying the engine kind and config, one can get a specific implementation.

## Reference

- [paddle_inference_api.h](./paddle_inference_api.h)
- [some demos](./demo_ci)
