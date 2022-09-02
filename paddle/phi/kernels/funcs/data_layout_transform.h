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

#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"  // NOLINT
#endif

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

#ifdef PADDLE_WITH_MKLDNN

using MKLDNNDataType = dnnl::memory::data_type;
using MKLDNNMemoryFormat = dnnl::memory::format_tag;

inline MKLDNNMemoryFormat ToMKLDNNFormat(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::NHWC:
      return MKLDNNMemoryFormat::nhwc;
    case DataLayout::NCHW:
      return MKLDNNMemoryFormat::nchw;
    case DataLayout::NCDHW:
      return MKLDNNMemoryFormat::ncdhw;
    case DataLayout::NDHWC:
      return MKLDNNMemoryFormat::ndhwc;
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "Fail to convert layout %s to MKLDNN format.",
          ::paddle::framework::DataLayoutToString(layout)));
  }
}

// Caution: proto::VarType::Type -> phi::DataType after transfer
inline MKLDNNDataType ToMKLDNNDataType(DataType type) {
  struct DataTypeHash {
    std::size_t operator()(const DataType& f) const {
      return std::hash<int>{}(static_cast<int>(f));
    }
  };
  struct DataTypeEqual {
    bool operator()(const DataType& lhs, const DataType& rhs) const {
      return static_cast<int>(lhs) == static_cast<int>(rhs);
    }
  };

  static std::
      unordered_map<DataType, MKLDNNDataType, DataTypeHash, DataTypeEqual>
          dict{{DataType::FLOAT32, MKLDNNDataType::f32},
               {DataType::INT8, MKLDNNDataType::s8},
               {DataType::UINT8, MKLDNNDataType::u8},
               {DataType::INT32, MKLDNNDataType::s32},
               {DataType::BFLOAT16, MKLDNNDataType::bf16}};
  auto iter = dict.find(type);
  if (iter != dict.end()) return iter->second;
  return MKLDNNDataType::undef;
}

void innerTransDataLayoutFromMKLDNN(DataLayout in_layout,
                                    DataLayout out_layout,
                                    const DenseTensor& in,
                                    DenseTensor* out,
                                    Place place,
                                    bool always_copy = false);
void* GetDataFromTensor(const DenseTensor& tensor, MKLDNNDataType type);

#endif

}  // namespace funcs
}  // namespace phi
