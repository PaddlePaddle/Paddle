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
  if (dims_size == 1) {
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
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Paddle support tensors with rank in range <1, 9>, but received "
          "tensor with rank: %d",
          tensor_rank));
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
    VLOG(3) << "Keeping MKLDNN/NHWC/NDHWC output_shape"
            << print_dims(phi::vectorize<int>(tensor_in->dims()));
    return;
  }

  switch (from) {
    case DataLayout::MKLDNN:
      if ((to == DataLayout::NHWC) || (to == DataLayout::NDHWC)) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: MKLDNN to: NHWC/NDHWC output_shape"
                << print_dims(dims);
      }
      break;
    case DataLayout::NHWC:
    case DataLayout::NDHWC:
      if (to == DataLayout::MKLDNN) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: NHWC/NDHWC to: MKLDNN output_shape"
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

}  // namespace funcs
}  // namespace phi
