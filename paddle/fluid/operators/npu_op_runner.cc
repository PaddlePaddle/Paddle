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

#include "paddle/fluid/operators/npu_op_runner.h"

#include <paddle/fluid/framework/data_type.h>
#include <paddle/fluid/framework/operator.h>

#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

static std::map<framework::proto::VarType::Type, aclDataType>
    DTYPE_2_ACL_DTYPE = {
        {framework::proto::VarType::BOOL, ACL_BOOL},
        {framework::proto::VarType::UINT8, ACL_UINT8},
        {framework::proto::VarType::INT8, ACL_INT8},
        {framework::proto::VarType::INT16, ACL_INT16},
        {framework::proto::VarType::INT32, ACL_INT32},
        {framework::proto::VarType::INT64, ACL_INT64},
        {framework::proto::VarType::FP16, ACL_FLOAT16},
        {framework::proto::VarType::FP32, ACL_FLOAT},
        {framework::proto::VarType::FP64, ACL_DOUBLE},
};

static std::map<DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
    {DataLayout::kNCHW, ACL_FORMAT_NCHW},
    {DataLayout::kNHWC, ACL_FORMAT_NHWC},
    {DataLayout::kAnyLayout, ACL_FORMAT_ND},
};

aclDataType ConvertToNpuDtype(framework::proto::VarType::Type dtype) {
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(iter, DTYPE_2_ACL_DTYPE.end(),
                    platform::errors::NotFound(
                        "The data type (%s) can not convert to ACL data type.",
                        framework::DataTypeToString(dtype)));
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

aclrtStream GetCurrentNPUStream(int device_id) {
  if (device_id == -1) {
    device_id = platform::GetCurrentNPUDeviceId();
  }
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx = static_cast<platform::NPUDeviceContext *>(
      pool.Get(platform::NPUPlace(device_id)));
  return dev_ctx->stream();
}

NpuOpRunner::NpuOpRunner() {}

NpuOpRunner::NpuOpRunner(const std::string &op_type) : op_type_(op_type) {}

NpuOpRunner::NpuOpRunner(const std::string &op_type,
                         const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         const NPUAttributeMap &attrs)
    : op_type_(op_type) {
  AddInputs(inputs);
  AddOutputs(outputs);
  AddAttrs(attrs);
}

NpuOpRunner::~NpuOpRunner() {
  VLOG(5) << "Free NpuOpRunner(" << this << ") of " << op_type_;
  // Is it safe to free the descs/buffers after run called in host ?
  aclopDestroyAttr(attr_);  // return void
  for (auto desc : input_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto desc : output_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto buffer : input_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
  for (auto buffer : output_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
}

const std::string &NpuOpRunner::Type() { return op_type_; }

NpuOpRunner &NpuOpRunner::SetType(const std::string &name) {
  op_type_ = name;
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttr(const std::string &name,
                                  const NPUAttribute &attr) {
  if (!attr_) {
    attr_ = aclopCreateAttr();
  }
  if (attr.type() == typeid(bool)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrBool(attr_, name.c_str(), BOOST_GET_CONST(bool, attr)));
  } else if (attr.type() == typeid(int)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int, attr)));

  } else if (attr.type() == typeid(int64_t)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int64_t, attr)));
  } else if (attr.type() == typeid(float)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrFloat(attr_, name.c_str(), BOOST_GET_CONST(float, attr)));
  } else if (attr.type() == typeid(std::vector<bool>)) {
    auto a = BOOST_GET_CONST(std::vector<bool>, attr);
    std::vector<uint8_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<uint8_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListBool(
        attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int>)) {
    auto a = BOOST_GET_CONST(std::vector<int>, attr);
    std::vector<int64_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<int64_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int64_t>)) {
    auto a = BOOST_GET_CONST(std::vector<int64_t>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::vector<float>)) {
    auto a = BOOST_GET_CONST(std::vector<float>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::string)) {
    auto a = BOOST_GET_CONST(std::string, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrString(attr_, name.c_str(), a.c_str()));
  } else if (attr.type() == typeid(std::vector<std::string>)) {
    auto a = BOOST_GET_CONST(std::vector<std::string>, attr);
    std::vector<const char *> s;
    for (auto &it : a) {
      s.push_back(it.data());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
  } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
    auto a = BOOST_GET_CONST(std::vector<std::vector<int64_t>>, attr);
    std::vector<int64_t *> data;
    std::vector<int> num;
    for (auto &&v : a) {
      data.push_back(v.data());
      num.push_back(v.size());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListListInt(
        attr_, name.c_str(), data.size(), num.data(), data.data()));
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Can not convert attribubte '%s' to convert to aclopAttr", name));
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrs(const NPUAttributeMap &attrs) {
  for (const auto &pair : attrs) {
    AddAttr(pair.first, pair.second);
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor, aclMemType mem_type) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor, mem_type));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(std::vector<int32_t> &&dims) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx =
      static_cast<platform::CPUDeviceContext *>(pool.Get(platform::CPUPlace()));
  Tensor host_tensor;
  TensorFromVector(dims, *dev_ctx, &host_tensor);
  host_tensors_.emplace_back(host_tensor);

  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(host_tensor, ACL_MEMTYPE_HOST));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(host_tensor));

  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(std::vector<int64_t> &&dims) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx =
      static_cast<platform::CPUDeviceContext *>(pool.Get(platform::CPUPlace()));
  Tensor host_tensor;
  TensorFromVector(dims, *dev_ctx, &host_tensor);
  host_tensors_.emplace_back(host_tensor);

  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(host_tensor, ACL_MEMTYPE_HOST));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(host_tensor));

  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(std::vector<float> &&values) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx =
      static_cast<platform::CPUDeviceContext *>(pool.Get(platform::CPUPlace()));
  Tensor host_tensor;
  TensorFromVector(values, *dev_ctx, &host_tensor);
  host_tensors_.emplace_back(host_tensor);

  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(host_tensor, ACL_MEMTYPE_HOST));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(host_tensor));

  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(std::vector<double> &&values) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx =
      static_cast<platform::CPUDeviceContext *>(pool.Get(platform::CPUPlace()));
  Tensor host_tensor;
  TensorFromVector(values, *dev_ctx, &host_tensor);
  host_tensors_.emplace_back(host_tensor);

  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(host_tensor, ACL_MEMTYPE_HOST));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(host_tensor));

  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutput(const Tensor &tensor) {
  // create aclTensorDesc
  output_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  output_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInputs(const std::vector<Tensor> &tensors) {
  input_descs_.reserve(tensors.size());
  input_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

// NOTE(zhiqiu): For operators whose input is a list (such as concat, stack),
// It is needed to set the name of each input tensor.
NpuOpRunner &NpuOpRunner::AddInputNames(const std::vector<std::string> &names) {
  PADDLE_ENFORCE_EQ(names.size(), input_descs_.size(),
                    platform::errors::InvalidArgument(
                        "The size of input names should be "
                        "equal to the size of input descs, but got the size "
                        "of input names is %d, the size of input descs is %d.",
                        names.size(), input_descs_.size()));
  for (size_t i = 0; i < names.size(); ++i) {
    aclSetTensorDescName(input_descs_[i], names[i].c_str());
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutputs(const std::vector<Tensor> &tensors) {
  output_descs_.reserve(tensors.size());
  output_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    output_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    output_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

aclTensorDesc *NpuOpRunner::GetInputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, input_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), index, input_descs_.size()));
  return input_descs_[index];
}

aclTensorDesc *NpuOpRunner::GetOutputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index, output_descs_.size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of output of "
                        "operator %s, but got index is %d and size is %d",
                        Type(), index, output_descs_.size()));
  return output_descs_[index];
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetInputDescs() {
  return input_descs_;
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetOutputDescs() {
  return output_descs_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetInputBuffers() {
  return input_buffers_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetOutputBuffers() {
  return output_buffers_;
}

aclTensorDesc *NpuOpRunner::CreateTensorDesc(Tensor tensor,
                                             aclMemType mem_type) {
  auto dtype = ConvertToNpuDtype(tensor.type());
  auto format = ConvertToNpuFormat(tensor.layout());
  auto dims = framework::vectorize(tensor.dims());
  int size = dims.size();
  // TODO(pangyoki): `keep_prob` used in `DropOutGenMask` NPU
  // OP must be a scalar with shape[0]. At present, the shape
  // of the `prob` Tensor of this OP is forced to be set to 0
  // in `npu_op_runner.cc`, which needs to be optimized later.
  if (op_type_ == "DropOutGenMask" && size == 1 && *(dims.data()) == 1) {
    size = 0;
  }

  VLOG(4) << "NPU dtype:" << dtype << " "
          << "rank:" << dims.size() << " dims:" << tensor.dims()
          << " format:" << format;

  auto *desc = aclCreateTensorDesc(dtype, size, dims.data(), format);
  PADDLE_ENFORCE_NOT_NULL(
      desc, platform::errors::External("Call aclCreateTensorDesc failed."));
  PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc, format));
  PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageShape(desc, size, dims.data()));
  if (mem_type == ACL_MEMTYPE_HOST) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorPlaceMent(desc, mem_type));
  }
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(Tensor tensor) {
  void *ptr = tensor.data<void>();
  VLOG(4) << "NPU ptr: " << ptr << ", size: " << tensor.memory_size();
  auto *buffer = aclCreateDataBuffer(ptr, tensor.memory_size());
  PADDLE_ENFORCE_NOT_NULL(
      buffer, platform::errors::External("Call aclCreateDataBuffer failed."));
  return buffer;
}

void NpuOpRunner::Run(aclrtStream stream) const {
  if (!stream) {
    VLOG(4) << "Run with default current npu stream: " << stream;
    stream = GetCurrentNPUStream();
  }
  VLOG(5) << "NpuOpRunner(" << this << ") Run:";
  VLOG(4) << "op_type: " << op_type_;
  VLOG(4) << "input_desc.size: " << input_descs_.size();
  VLOG(4) << "output_desc.size: " << output_descs_.size();
  VLOG(4) << "attr: " << attr_;
  VLOG(4) << "stream: " << stream;

  aclError ret = aclopCompileAndExecute(
      op_type_.c_str(), input_descs_.size(), input_descs_.data(),
      input_buffers_.data(), output_descs_.size(), output_descs_.data(),
      output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
      stream);
  VLOG(4) << "after aclopCompileAndExecute: " << ret;
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
}

}  // namespace operators
}  // namespace paddle
