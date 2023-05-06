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

using OneDNNDataType = dnnl::memory::data_type;
using OneDNNMemoryFormat = dnnl::memory::format_tag;

inline OneDNNMemoryFormat ToOneDNNFormat(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::NHWC:
      return OneDNNMemoryFormat::nhwc;
    case DataLayout::NCHW:
      return OneDNNMemoryFormat::nchw;
    case DataLayout::NCDHW:
      return OneDNNMemoryFormat::ncdhw;
    case DataLayout::NDHWC:
      return OneDNNMemoryFormat::ndhwc;
    default:
      PADDLE_THROW(
          errors::InvalidArgument("Fail to convert layout %s to oneDNN format.",
                                  ::phi::DataLayoutToString(layout)));
  }
}

inline OneDNNDataType ToOneDNNDataType(DataType type) {
#if __GNUC__ > 5
  using DataTypeMapping = std::unordered_map<DataType, OneDNNDataType>;
#else
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
  using DataTypeMapping =
      std::unordered_map<DataType, OneDNNDataType, DataTypeHash, DataTypeEqual>;
#endif

  static DataTypeMapping dict{{DataType::FLOAT32, OneDNNDataType::f32},
                              {DataType::INT8, OneDNNDataType::s8},
                              {DataType::UINT8, OneDNNDataType::u8},
                              {DataType::INT32, OneDNNDataType::s32},
                              {DataType::BFLOAT16, OneDNNDataType::bf16}};

  auto iter = dict.find(type);
  if (iter != dict.end()) return iter->second;
  return OneDNNDataType::undef;
}

void TransDataLayoutFromOneDNN(DataLayout in_layout,
                               DataLayout out_layout,
                               const DenseTensor& in,
                               DenseTensor* out,
                               Place place,
                               bool always_copy = false);
void* GetDataFromTensor(const DenseTensor& tensor, OneDNNDataType type);

dnnl::memory::desc make_memory_desc(const phi::DenseTensor& ref_tensor,
                                    phi::DataLayout target_layout);

#endif

}  // namespace funcs
}  // namespace phi
