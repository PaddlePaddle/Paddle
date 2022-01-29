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
#include <popart/vendored/any.hpp>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {
namespace ipu {

using float16 = platform::float16;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using Scope = framework::Scope;
using OpDesc = framework::OpDesc;
using Graph = framework::ir::Graph;
using Node = framework::ir::Node;
using BlockDesc = framework::BlockDesc;

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
  explicit PaddleIArray(const Tensor* tensor) {
    tensor_.ShareDataWith(*tensor);
    for (int i = 0; i < tensor->dims().size(); ++i) {
      shape_.push_back(tensor->dims().at(i));
    }
  }

 public:
  void* data();
  popart::DataType dataType() const;
  std::size_t rank() const;
  int64_t dim(size_t index) const;
  std::size_t nelms() const;
  const popart::Shape shape() const;

 private:
  Tensor tensor_;
  std::vector<int64_t> shape_;
};

popart::DataType VarType2PopartType(const framework::proto::VarType::Type type);
framework::proto::VarType::Type PopartType2VarType(const popart::DataType type);
popart::DataType OnnxDtype2PopartType(const int type);
bool GetBoolEnv(std::string str);

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> Tensor2IArray(const Tensor& tensor) {
  auto dtype = VarType2PopartType(tensor.type());
  auto shape = std::vector<int64_t>();
  for (size_t i = 0; i < tensor.dims().size(); ++i) {
    shape.push_back(tensor.dims().at(i));
  }
  popart::TensorInfo tensor_info(dtype, shape);

  return std::make_unique<popart::NDArrayWrapper<T>>(
      reinterpret_cast<T*>(tensor.data()), tensor_info);
}

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> LoDTensor2IArray(
    LoDTensor const& lod_tensor) {
  if (lod_tensor.lod().size() == 0) {
    return Tensor2IArray<T>(lod_tensor);
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("LoDTensor2IArray is Unimplemented"));
  }
}

template <typename T>
T GetSingleVarFromScope(const Scope* scope, const std::string& var_name) {
  auto var = scope->GetVar(var_name);
  auto tensor = var->Get<framework::LoDTensor>();
  // check dtype is  ?
  return tensor.data<T>()[0];
}

struct CustomOpAttrVisitor : public boost::static_visitor<void> {
  explicit CustomOpAttrVisitor(std::map<std::string, popart::any>* attr,
                               const std::string& attr_name)
      : attrs_(attr), attr_name_(attr_name) {}
  mutable std::map<std::string, popart::any>* attrs_;
  std::string attr_name_;

  void operator()(int v) const { attrs_->emplace(attr_name_, v); }
  void operator()(float v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::string& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<int>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<float>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<std::string>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(bool v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::vector<bool>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(BlockDesc* desc) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `BlockDesc` type."));
  }
  void operator()(const std::vector<BlockDesc*>& v) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `BlockDesc` type."));
  }
  void operator()(int64_t v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::vector<int64_t>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<double>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(boost::blank) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `boost::blank` type."));
  }
};

struct IpuCustomOpIdentifier {
  IpuCustomOpIdentifier(const std::string& _paddle_op,
                        const std::string& _popart_op,
                        const std::string& _domain, unsigned int _version)
      : paddle_op(_paddle_op), popart_op(_domain, _popart_op, _version) {}

  std::string repr() {
    std::ostringstream os;
    os << "paddle_op: " << paddle_op << ", domain: " << popart_op.domain
       << ", type: " << popart_op.type << ", version: " << popart_op.version;
    return os.str();
  }

  std::string paddle_op;
  popart::OperatorIdentifier popart_op;
};

struct ConstantOpAttrVisitor : public boost::static_visitor<void> {
  explicit ConstantOpAttrVisitor(framework::LoDTensor* tensor,
                                 framework::proto::VarType::Type dtype)
      : tensor_(tensor), dtype_(dtype) {}
  framework::LoDTensor* tensor_;
  framework::proto::VarType::Type dtype_;

  void operator()(const std::vector<int>& vec) const {
    framework::TensorFromVector<int>(vec, tensor_);
  }
  void operator()(const std::vector<float>& vec) const {
    if (dtype_ == framework::proto::VarType::FP16) {
      std::vector<float16> vec_fp16;
      std::transform(vec.begin(), vec.end(), std::back_inserter(vec_fp16),
                     [](float f) -> float16 { return float16(f); });
      framework::TensorFromVector<float16>(vec_fp16, tensor_);
    } else {
      framework::TensorFromVector<float>(vec, tensor_);
    }
  }
  void operator()(const std::vector<bool>& vec) const {
    framework::TensorFromVector<bool>(vec, tensor_);
  }
  void operator()(const std::vector<int64_t>& vec) const {
    framework::TensorFromVector<int64_t>(vec, tensor_);
  }
  void operator()(const std::vector<double>& vec) const {
    framework::TensorFromVector<double>(vec, tensor_);
  }
  void RaiseError() const {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Constant value must be a vector"));
  }
  void operator()(int v) const { RaiseError(); }
  void operator()(float v) const { RaiseError(); }
  void operator()(const std::string& v) const { RaiseError(); }
  void operator()(const std::vector<std::string>& v) const { RaiseError(); }
  void operator()(bool v) const { RaiseError(); }
  void operator()(BlockDesc* desc) const { RaiseError(); }
  void operator()(const std::vector<BlockDesc*>& v) const { RaiseError(); }
  void operator()(int64_t v) const { RaiseError(); }
  void operator()(boost::blank) const { RaiseError(); }
};

std::vector<std::pair<std::string, std::string>> GetOptPrePostfix(
    const std::string& opt_type);

int RequestIpus(const int num_ipus);

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
