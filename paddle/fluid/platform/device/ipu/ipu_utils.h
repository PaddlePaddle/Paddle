/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <popart/ndarraywrapper.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace platform {
namespace ipu {

// onnx dtype
// https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3
enum ONNXDataType : int {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16
};

class PaddleIArray final : public popart::IArray {
 public:
  explicit PaddleIArray(framework::Tensor *tensor) : tensor_(tensor) {
    for (int i = 0; i < tensor->dims().size(); ++i) {
      shape_.push_back(tensor->dims().at(i));
    }
  }

 public:
  void *data();
  popart::DataType dataType() const;
  std::size_t rank() const;
  int64_t dim(size_t index) const;
  std::size_t nelms() const;
  const popart::Shape shape() const;

 private:
  framework::Tensor *tensor_;
  std::vector<int64_t> shape_;
};

popart::DataType VarType2PopartType(const framework::proto::VarType::Type type);
framework::proto::VarType::Type PopartType2VarType(const popart::DataType type);
popart::DataType OnnxDtype2PopartType(const int type);
bool GetBoolEnv(std::string str);

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> Tensor2IArray(
    const framework::Tensor &tensor) {
  auto dtype = VarType2PopartType(tensor.type());
  auto shape = std::vector<int64_t>();
  for (size_t i = 0; i < tensor.dims().size(); ++i) {
    shape.push_back(tensor.dims().at(i));
  }
  popart::TensorInfo tensor_info(dtype, shape);

  return std::make_unique<popart::NDArrayWrapper<T>>(
      reinterpret_cast<T *>(tensor.data<void>()), tensor_info);
}

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> LoDTensor2IArray(
    framework::LoDTensor const &lod_tensor) {
  PADDLE_ENFORCE_EQ(
      lod_tensor.lod().size(), 0UL,
      platform::errors::InvalidArgument("LoDTensor2IArray is Unimplemented"));
  return Tensor2IArray<T>(lod_tensor);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
