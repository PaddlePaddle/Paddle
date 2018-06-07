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
#include <vector>
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_MKLDNN
using MKLDNNFormat = mkldnn::memory::format;
using MKLDNNDataType = mkldnn::memory::data_type;

inline MKLDNNFormat ToMKLDNNFormat(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::kNHWC:
      return MKLDNNFormat::nhwc;
    case DataLayout::kNCHW:
      return MKLDNNFormat::nchw;
    default:
      PADDLE_THROW("Fail to convert layout %s to MKLDNN format",
                   DataLayoutToString(layout));
  }
}

inline DataLayout ToPaddleLayout(const MKLDNNFormat& format) {
  switch (format) {
    case MKLDNNFormat::nhwc:
      return DataLayout::kNHWC;
    case MKLDNNFormat::nchw:
      return DataLayout::kNCHW;
    default:
      PADDLE_THROW("Fail to convert MKLDNN format to paddle layout");
  }
}

inline MKLDNNDataType ToMKLDNNDataType(const std::type_index type) {
  static const std::map<std::type_index, MKLDNNDataType> dict{
      {std::type_index(typeid(float)), MKLDNNDataType::f32},  // NOLINT
      {std::type_index(typeid(char)), MKLDNNDataType::s8},    // NOLINT
      {std::type_index(typeid(unsigned char)), MKLDNNDataType::u8},
      {std::type_index(typeid(int16_t)), MKLDNNDataType::s16},
      {std::type_index(typeid(int32_t)), MKLDNNDataType::s32}};
  auto iter = dict.find(type);
  if (iter != dict.end()) return iter->second;
  return MKLDNNDataType::data_undef;
}
#endif

void TransDataLayoutFromMKLDNN(const OpKernelType& kernel_type_for_var,
                               const OpKernelType& expected_kernel_type,
                               const Tensor& in, Tensor* out);

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to);

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type, const Tensor& in,
                     Tensor* out);

}  // namespace framework
}  // namespace paddle
