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
#include "paddle/fluid/operators/npu_op_runner.h"

#include <paddle/fluid/framework/operator.h>

#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

static std::map<proto::VarType::Type, aclDataType> DTYPE_2_ACL_DTYPE = {
    {proto::VarType::BOOL, ACL_BOOL},    {proto::VarType::INT16, ACL_INT16},
    {proto::VarType::INT32, ACL_INT32},  {proto::VarType::INT64, ACL_INT64},
    {proto::VarType::FP16, ACL_FLOAT16}, {proto::VarType::FP32, ACL_FLOAT},
    {proto::VarType::FP64, ACL_DOUBLE},
};

static std::map<DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
    {DataLayout::kNCHW, ACL_FORMAT_NCHW},
    {DataLayout::kNHWC, ACL_FORMAT_NHWC},
    {DataLayout::kAnyLayout, ACL_FORMAT_ND},
};

aclDataType ConvertToNpuDtype(proto::VarType::Type dtype) {
  auto iter = PADDLE_DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(iter, PADDLE_DTYPE_2_ACL_DTYPE.end(),
                    platform::errors::NotFound(
                        "The data type (%s) can not convert to ACL data type.",
                        DataTypeTostring(dtype)));
  return iter->second;
}

aclFormat ConvertToNpuFormat(DataLayout layout) {
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter, DATA_LAYOUT_2_ACL_FORMAT.end(),
      platform::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

aclFormat ConvertToNpuAttr(Attribute attr) {
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter, DATA_LAYOUT_2_ACL_FORMAT.end(),
      platform::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

NpuOpRunner::NpuOpRunner(string op_type) : op_type_(op_type) {}
NpuOpRunner::NpuOpRunner(string op_type, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         const AttributeMap &attrs)
    : op_type_(op_type) {
  AddInputs(inputs);
  AddOutputs(outputs);
  AddAttrs(attrs);
}

const string &NpuOpRunner::Type() { return op_type_; }

NpuOpRunner &NpuOpRunner::AddAttr(const std::string &name,
                                  const Attribute &attr) {
  switch (attr.type()) {
    case typeid(bool):  // NOLINT
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrBool(attr_, name.c_str(), BOOST_GET_CONST(bool, attr)));
      break;
    case typeid(int):  // NOLINT
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int, attr)));
      break;
    case typeid(int64_t):  // NOLINT
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(),
                          static_cast<int>(BOOST_GET_CONST(int64_t, attr))));
      VLOG(4) << "Downcast attribute (" << name << ") from int64 (" <<
        BOOST_GET_CONST(int64_t, attr) << ") to int (" <<
        static_cast<int>(BOOST_GET_CONST(int64_t, attr))) << ")";
      break;
    case typeid(float):  // NOLINT
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrFloat(attr_, name.c_str(), BOOST_GET_CONST(float, attr)));
      break;
    case typeid(std::vector<bool>):
      auto a = BOOST_GET_CONST(std::vector<bool>, attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListBool(attr_, name.c_str(), a.size(), a.data()));
      break;
    case typeid(std::vector<int>):
      auto a = BOOST_GET_CONST(std::vector<int>, attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
      break;
    case typeid(std::vector<float>):
      auto a = BOOST_GET_CONST(std::vector<float>, attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
      break;
    case typeid(std::string):
      auto a = BOOST_GET_CONST(std::string, attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListFloat(attr_, name.c_str(), a.c_str()));
      break;
    case typeid(std::vector<std::string>):
      auto a = BOOST_GET_CONST(std::vector<std::string>, attr);
      std::vector<char *> s;
      for (auto &it : a) {
        s.push_back(const_cast<char *>(it.data()));
      }
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListFloat(attr_, name.c_str(), s.size(), s.data()));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Can not convert attribubte '%s' to convert to aclopAttr", name));
  }
}

NpuOpRunner &NpuOpRunner::AddAttrs(const AttributeMap &attrs) {
  for (const auto &pair : attrs) {
    AddAttr(pair.first, pair.second);
  }
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  input_buffers.emplace_back(CreateDataBuffer(tensor));
}

NpuOpRunner &NpuOpRunner::AddOutput(const Tensor &tensor) {
  // create aclTensorDesc
  output_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  output_buffers.emplace_back(CreateDataBuffer(tensor));
}

NpuOpRunner &NpuOpRunner::AddInputs(const std::vector<Tensor> &tensors) {
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers.emplace_back(CreateDataBuffer(tensor));
  }
}

NpuOpRunner &NpuOpRunner::AddOutputs(const std::vector<Tensor> &tensors) {
  for (auto tensor : tensors) {
    // create aclTensorDesc
    output_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    output_buffers.emplace_back(CreateDataBuffer(tensor));
  }
}

aclTensorDesc *NpuOpRunner::GetInputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, input_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), idx, input_descs_.size()));
  return input_descs_[index];
}

aclTensorDesc *NpuOpRunner::GetOutputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, output_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of output of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), idx, output_descs_.size()));
  return output_descs_[index];
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetInputDescs() {
  return input_descs_;
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetOutputDescs() {
  return output_descs_;
}

std::vector<aclDataBuffer *> &GetInputBuffers() { return input_buffers_; }

std::vector<aclDataBuffer *> &GetOutputBuffers() { return output_buffers_; }

aclTensorDesc *NpuOpRunner::CreateTensorDesc(const Tensor &tensor) {
  auto dtype = ConvertToNpuDtype(tensor.type());
  auto format = ConvertToNpuDtype(tensor.layout());
  auto dims = tensor.layout();

  auto *desc = aclCreateTensorDesc(dtype, dims.size(), dims.data(), format);
  PADDLE_ENFORCE_NOT_NULL(
      desc, platform::errors::External("Call aclCreateTensorDesc failed."));
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(const Tensor &tensor) {
  auto *buffer =
      aclCreateDataBuffer(tensor.data<uint8_t>(), tensor.memory_size());
  PADDLE_ENFORCE_NOT_NULL(
      buffer, platform::errors::External("Call aclCreateDataBuffer failed."));
  return buffer;
}

void NpuOpRunner::Run(aclrtStream stream) {
  aclError ret = aclopExecuteV2(op_type_, input_descs_.size(),
                                input_descs_.data(), input_buffers_.data(),
                                output_descs_.size(), output_descs_.data(),
                                output_buffers_.data(), attr_, stream);
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
}
}  // namespace operators
}  // namespace paddle
