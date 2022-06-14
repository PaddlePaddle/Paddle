// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

#include "glog/logging.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace tensorrt {

static nvinfer1::DataType TensorTypeToWeightType(::phi::DataType tensor_type) {
  switch (tensor_type) {
    case ::phi::DataType::FLOAT32:
      return nvinfer1::DataType::kFLOAT;
    case ::phi::DataType::INT32:
      return nvinfer1::DataType::kINT32;
    case ::phi::DataType::FLOAT16:
      return nvinfer1::DataType::kHALF;
    default:
      llvm_unreachable("should not reach here");
  }
}

static nvinfer1::Dims ArrayAttrToNvDims(const mlir::ArrayAttr& int_array_attr) {
  nvinfer1::Dims dims;
  dims.nbDims = int_array_attr.size();
  CHECK(!int_array_attr.empty());
  CHECK(int_array_attr[0].getType().isIntOrIndex());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = int_array_attr[i].cast<mlir::IntegerAttr>().getInt();
  }
  return dims;
}

template <typename T>
static std::vector<T> ArrayAttrToVec(const mlir::ArrayAttr& int_array_attr) {
  std::vector<T> ret;
  ret.resize(int_array_attr.size());
  CHECK(!int_array_attr.empty());
  CHECK(int_array_attr[0].getType().isIntOrIndex());
  for (size_t i = 0; i < int_array_attr.size(); ++i) {
    ret[i] = int_array_attr[i].cast<mlir::IntegerAttr>().getInt();
  }
  return ret;
}

static nvinfer1::Weights TensorToWeights(::phi::DenseTensor* tensor) {
  CHECK_NOTNULL(tensor);
  nvinfer1::Weights ret;
  ret.type = TensorTypeToWeightType(tensor->dtype());
  ret.count = tensor->numel();
  ret.values = tensor->data();
  return ret;
}

}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
