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

#include <thread>
#include "dnnl.hpp"  // NOLINT
#include "glog/logging.h"

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

using OneDNNMemoryFormat = dnnl::memory::format_tag;
using OneDNNDataType = dnnl::memory::data_type;

template <typename Type>
void* to_void_cast(const Type* t) {
  return static_cast<void*>(const_cast<Type*>(t));
}

inline OneDNNMemoryFormat OneDNNFormatForSize(size_t dims_size,
                                              OneDNNMemoryFormat data_format) {
  if (dims_size == 0) {
    return OneDNNMemoryFormat::x;
  } else if (dims_size == 1) {
    return OneDNNMemoryFormat::x;
  } else if (dims_size == 2) {
    return OneDNNMemoryFormat::nc;
  } else if (dims_size == 3) {
    if (data_format == OneDNNMemoryFormat::nchw) {
      return OneDNNMemoryFormat::ncw;
    } else if (data_format == OneDNNMemoryFormat::nhwc) {
      return OneDNNMemoryFormat::nwc;
    }
  } else if (dims_size == 4) {
    if (data_format == OneDNNMemoryFormat::goihw) {
      return OneDNNMemoryFormat::oihw;
    }
  } else if (dims_size == 5) {
    if (data_format == OneDNNMemoryFormat::goidhw) {
      return OneDNNMemoryFormat::oidhw;
    }
    if (data_format == OneDNNMemoryFormat::nchw) {
      return OneDNNMemoryFormat::ncdhw;
    } else if (data_format == OneDNNMemoryFormat::nhwc) {
      return OneDNNMemoryFormat::ndhwc;
    }
  } else if (dims_size == 6) {
    if (data_format == OneDNNMemoryFormat::nchw) {
      return OneDNNMemoryFormat::abcdef;
    }
  }
  return data_format;
}

inline dnnl::memory::format_tag GetPlainOneDNNFormat(int tensor_rank) {
  switch (tensor_rank) {
    case 0:
      // use 1D to represent 0D
      return dnnl::memory::format_tag::a;
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    case 7:
      return dnnl::memory::format_tag::abcdefg;
    case 8:
      return dnnl::memory::format_tag::abcdefgh;
    case 9:
      return dnnl::memory::format_tag::abcdefghi;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Paddle support tensors with rank in range <1, 9>, but received "
          "tensor with rank: %d",
          tensor_rank));
  }
}

template <typename Type>
dnnl::memory::data_type OneDNNGetDataType() {
  return dnnl::memory::data_type::undef;
}

template <>
inline dnnl::memory::data_type OneDNNGetDataType<float>() {
  return dnnl::memory::data_type::f32;
}
template <>
inline dnnl::memory::data_type OneDNNGetDataType<int32_t>() {
  return dnnl::memory::data_type::s32;
}
template <>
inline dnnl::memory::data_type OneDNNGetDataType<int8_t>() {
  return dnnl::memory::data_type::s8;
}
template <>
inline dnnl::memory::data_type OneDNNGetDataType<uint8_t>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDNNGetDataType<dtype::bfloat16>() {
  return dnnl::memory::data_type::bf16;
}

inline std::vector<std::vector<int64_t>> ToOneDNNPadding(
    const std::vector<int64_t>& paddings) {
  if (paddings.size() == 6) {
    int padding_front = paddings[0];
    int padding_back = paddings[1];
    int padding_top = paddings[2];
    int padding_bottom = paddings[3];
    int padding_left = paddings[4];
    int padding_right = paddings[5];

    return {{padding_front, padding_top, padding_left},
            {padding_back, padding_bottom, padding_right}};
  } else {
    int padding_top = paddings[0];
    int padding_bottom = paddings[1];
    int padding_left = paddings[2];
    int padding_right = paddings[3];

    return {{padding_top, padding_left}, {padding_bottom, padding_right}};
  }
}

template <typename T>
inline void AppendKey(std::string* key, const T& num) {
  key->append(std::to_string(num));
}

template <>
inline void AppendKey(std::string* key,
                      const dnnl::memory::format_tag& format) {
  key->append(std::to_string(static_cast<int>(format)));
}

template <>
inline void AppendKey(std::string* key,
                      const dnnl::memory::data_type& data_type) {
  key->append(std::to_string(static_cast<int>(data_type)));
}

template <>
inline void AppendKey(std::string* key, const dnnl::algorithm& algorithm) {
  key->append(std::to_string(static_cast<int>(algorithm)));
}

template <>
inline void AppendKey(std::string* key,
                      const dnnl::normalization_flags& flags) {
  key->append(std::to_string(static_cast<int>(flags)));
}

inline void AppendKey(std::string* key, const std::string& str) {
  key->append(str);
}

inline void AppendKey(std::string* key, const char* str) { key->append(str); }

template <typename T>
inline void AppendKey(std::string* key, const std::vector<T>& dims) {
  for (size_t i = 0; i < dims.size(); i++) {
    AppendKey(key, std::to_string(dims[i]));
  }
}

template <typename... ArgTypes>
inline std::string CreateKey(const OneDNNContext& dev_ctx UNUSED,
                             ArgTypes&&... args) {
  std::string key;
  key.reserve(64);
  using expand_type = int[];
  expand_type{0, (AppendKey(&key, std::forward<ArgTypes>(args)), 0)...};
  key += OneDNNContext::tls().get_key_suffix();
  return key;
}

// The function adjusts the vector of weight dimensions for group convolutions
inline void GetGroupConvWeightsTz(std::vector<int64_t>& weights_tz,  // NOLINT
                                  const int groups) {
  if (groups > 1) {
    // if (is_conv3d) [o, i, d, h, w]->[g, o/g, i, d, h, w]
    // else [o, i, h, w] -> [g, o/g, i, h, w]
    weights_tz.push_back(0);
    std::rotate(weights_tz.begin(), weights_tz.end() - 1, weights_tz.end());
    weights_tz[0] = groups;
    weights_tz[1] = weights_tz[1] / groups;
  }
}

inline void MatchShapeToLayout(DenseTensor* tensor_in,
                               DataLayout from,
                               DataLayout to) {
  auto print_dims = [](const std::vector<int>& dims) {
    std::ostringstream oss;

    if (!dims.empty()) {
      oss << "[";
      // Convert all but the last element to avoid a trailing ","
      std::copy(
          dims.begin(), dims.end() - 1, std::ostream_iterator<int>(oss, ","));

      // Now add the last element with no delimiter
      oss << dims.back() << "]";
    }

    return oss.str();
  };

  // In these data layouts, channel dimension is either on 2nd position: nChw or
  // at last nhwC, so for dim==2 these layouts are the same and nothing should
  // be done. Similarly for dim==1 when you have just one possible combination.
  if (tensor_in->dims().size() < 3) {
    VLOG(3) << "Keeping ONEDNN/NHWC/NDHWC output_shape"
            << print_dims(phi::vectorize<int>(tensor_in->dims()));
    return;
  }

  switch (from) {
    case DataLayout::ONEDNN:
      if ((to == DataLayout::NHWC) || (to == DataLayout::NDHWC)) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: ONEDNN to: NHWC/NDHWC output_shape"
                << print_dims(dims);
      }
      break;
    case DataLayout::NHWC:
    case DataLayout::NDHWC:
      if (to == DataLayout::ONEDNN) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: NHWC/NDHWC to: ONEDNN output_shape"
                << print_dims(dims);
      }
      break;
    default:
      break;
  }
}

struct onednn_dummy_primitive {
  struct primitive_desc {};
  struct desc {};
};

inline dnnl::memory::desc OneDNNMemDesc(const std::vector<int64_t>& dims,
                                        dnnl::memory::data_type data_type,
                                        OneDNNMemoryFormat format) {
  return dnnl::memory::desc({dims}, data_type, format);
}

inline std::string ThreadIDasStr(void) {
  return std::to_string(
      std::hash<std::thread::id>()(std::this_thread::get_id()));
}

inline std::string ExtendKeyWithThreadInfoIfNeeded(
    const OneDNNContext& dev_ctx UNUSED, const std::string& key) {
  return (OneDNNContext::tls().is_tid_used_in_key() == true)
             ? key + "-t:" + ThreadIDasStr()
             : key;
}

enum class RNNReorderType { PP_NTC, PP_TNC, NTC_PP, TNC_PP };

}  // namespace funcs
}  // namespace phi
