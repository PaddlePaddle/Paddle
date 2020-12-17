//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
class OpKernelType;
class Tensor;
}  // namespace framework
}  // namespace paddle

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_MKLDNN
using MKLDNNDataType = mkldnn::memory::data_type;

inline MKLDNNMemoryFormat ToMKLDNNFormat(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::kNHWC:
      return MKLDNNMemoryFormat::nhwc;
    case DataLayout::kNCHW:
      return MKLDNNMemoryFormat::nchw;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Fail to convert layout %s to MKLDNN format.",
          DataLayoutToString(layout)));
  }
}

inline DataLayout ToPaddleLayout(const MKLDNNMemoryFormat& format) {
  switch (format) {
    case MKLDNNMemoryFormat::nhwc:
      return DataLayout::kNHWC;
    case MKLDNNMemoryFormat::nchw:
      return DataLayout::kNCHW;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Fail to convert MKLDNN format to paddle layout."));
  }
}

inline MKLDNNDataType ToMKLDNNDataType(proto::VarType::Type type) {
  static std::unordered_map<int, MKLDNNDataType> dict{
      {DataTypeTrait<float>::DataType(), MKLDNNDataType::f32},
      {DataTypeTrait<int8_t>::DataType(), MKLDNNDataType::s8},
      {DataTypeTrait<uint8_t>::DataType(), MKLDNNDataType::u8},
      {DataTypeTrait<int32_t>::DataType(), MKLDNNDataType::s32},
      {DataTypeTrait<platform::bfloat16>::DataType(), MKLDNNDataType::bf16}};
  auto iter = dict.find(static_cast<int>(type));
  if (iter != dict.end()) return iter->second;
  return MKLDNNDataType::undef;
}

void innerTransDataLayoutFromMKLDNN(DataLayout in_layout, DataLayout out_layout,
                                    const Tensor& in, Tensor* out,
                                    platform::Place place);

void TransDataLayoutFromMKLDNN(const OpKernelType& kernel_type_for_var,
                               const OpKernelType& expected_kernel_type,
                               const Tensor& in, Tensor* out);

void* GetDataFromTensor(const Tensor& tensor, MKLDNNDataType type);

#endif

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to);

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type, const Tensor& in,
                     Tensor* out);

}  // namespace framework
}  // namespace paddle
