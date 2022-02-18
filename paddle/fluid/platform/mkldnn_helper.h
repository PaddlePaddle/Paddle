/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "dnnl.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
#ifdef PADDLE_WITH_MKLDNN
using MKLDNNMemoryFormat = dnnl::memory::format_tag;
#endif
namespace platform {

using MKLDNNStream = dnnl::stream;
using MKLDNNEngine = dnnl::engine;
using MKLDNNMemory = dnnl::memory;
using MKLDNNMemoryDescriptor = dnnl::memory::desc;
using MKLDNNPrimitive = dnnl::primitive;
using MKLDNNPrimitiveDesc = dnnl::handle<dnnl_primitive_desc_t>;

typedef std::unique_ptr<MKLDNNStream> MKLDNNStreamPtr;
typedef std::unique_ptr<MKLDNNEngine> MKLDNNEnginePtr;
typedef std::unique_ptr<MKLDNNMemory> MKLDNNMemoryPtr;
typedef std::unique_ptr<MKLDNNPrimitive> MKLDNNPrimitivePtr;
typedef std::unique_ptr<MKLDNNPrimitiveDesc> MKLDNNPrimitiveDescPtr;

template <typename Type>
void* to_void_cast(const Type* t) {
  return static_cast<void*>(const_cast<Type*>(t));
}

template <typename Type>
void* to_void_reinterpret_cast(const Type* t) {
  return reinterpret_cast<void*>(const_cast<Type*>(t));
}

template <class Type>
using tf_desc = typename Type::desc;

template <class Type>
using tf_pd = typename Type::primitive_desc;

template <typename Type, typename Engine, typename... Args>
std::shared_ptr<tf_pd<Type>> MKLDNNFwdPrimitiveDesc(const Engine& e,
                                                    Args&&... args) {
  auto desc = tf_desc<Type>(dnnl::prop_kind::forward, (args)...);
  auto pd = new tf_pd<Type>(desc, e);
  return std::shared_ptr<tf_pd<Type>>(pd);
}

template <typename Type, typename Engine, typename Primitive, typename... Args>
tf_pd<Type> MKLDNNBwdPrimitiveDesc(const Engine& e, const Primitive& p,
                                   Args&&... args) {
  auto desc = tf_desc<Type>(args...);
  return tf_pd<Type>(desc, e, p);
}

inline void MatchShapeToLayout(framework::Tensor* tensor_in,
                               framework::DataLayout from,
                               framework::DataLayout to) {
  // In these data layouts, channel dimension is either on 2nd position: nChw or
  // at last nhwC, so for dim==2 these layouts are the same and nothing should
  // be done. Similarly for dim==1 when you have just one possible combination.
  if (tensor_in->dims().size() < 3) {
    return;
  }

  auto print_dims = [](const std::vector<int>& dims) {
    std::ostringstream oss;

    if (!dims.empty()) {
      oss << "[";
      // Convert all but the last element to avoid a trailing ","
      std::copy(dims.begin(), dims.end() - 1,
                std::ostream_iterator<int>(oss, ","));

      // Now add the last element with no delimiter
      oss << dims.back() << "]";
    }

    return oss.str();
  };

  switch (from) {
    case framework::DataLayout::kMKLDNN:
      if (to == framework::DataLayout::kNHWC) {
        auto dims = framework::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
        tensor_in->Resize(framework::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kMKLDNN to: kNHWC output_shape"
                << print_dims(dims);
      }
      break;
    case framework::DataLayout::kNHWC:
      if (to == framework::DataLayout::kMKLDNN) {
        auto dims = framework::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
        tensor_in->Resize(framework::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kNHWC to: kMKLDNN output_shape"
                << print_dims(dims);
      }
      break;
    default:
      break;
  }
}

struct mkldnn_dummy_primitive {
  struct primitive_desc {};
  struct desc {};
};

inline dnnl::memory::desc MKLDNNMemDesc(const std::vector<int64_t>& dims,
                                        dnnl::memory::data_type data_type,
                                        MKLDNNMemoryFormat format) {
  return dnnl::memory::desc({dims}, data_type, format);
}

inline void ClearMKLDNNCache(const platform::Place& place,
                             void* ptr = nullptr) {
  // Clear mkl-dnn cache,
  if (platform::is_cpu_place(place)) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::MKLDNNDeviceContext* dev_ctx =
        (platform::MKLDNNDeviceContext*)pool.Get(place);
    dev_ctx->ResetBlobMap(ptr);
    platform::MKLDNNDeviceContext::tls().set_cur_paddle_data_layout(
        paddle::framework::DataLayout::kNCHW);
  }
}

inline void DontClearMKLDNNCache(const platform::Place& place) {
  // Clear mkl-dnn cache,
  if (platform::is_cpu_place(place)) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::MKLDNNDeviceContext* dev_ctx =
        (platform::MKLDNNDeviceContext*)pool.Get(place);
    dev_ctx->BlockNextCacheClearing();
  }
}

template <typename Type>
dnnl::memory::data_type MKLDNNGetDataType() {
  return dnnl::memory::data_type::undef;
}

template <>
inline dnnl::memory::data_type MKLDNNGetDataType<float>() {
  return dnnl::memory::data_type::f32;
}
template <>
inline dnnl::memory::data_type MKLDNNGetDataType<int32_t>() {
  return dnnl::memory::data_type::s32;
}
template <>
inline dnnl::memory::data_type MKLDNNGetDataType<int8_t>() {
  return dnnl::memory::data_type::s8;
}
template <>
inline dnnl::memory::data_type MKLDNNGetDataType<uint8_t>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type MKLDNNGetDataType<paddle::platform::bfloat16>() {
  return dnnl::memory::data_type::bf16;
}

inline void Reorder(dnnl::memory src, dnnl::memory dst,
                    const dnnl::engine& engine) {
  auto reorder_prim = dnnl::reorder(src, dst);
  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  platform::RecordEvent record_reorder("int_reorder",
                                       platform::EventRole::kUniqueOp);
  reorder_prim.execute(astream, src, dst);
  astream.wait();
}

inline dnnl::memory::format_tag GetMKLDNNFormat(dnnl::memory::desc mem_desc) {
  auto ndims = mem_desc.data.ndims;
  auto strides = mem_desc.data.format_desc.blocking.strides;
  auto inner_nblks = mem_desc.data.format_desc.blocking.inner_nblks;
  auto inner_blks = mem_desc.data.format_desc.blocking.inner_blks;
  auto inner_idxs = mem_desc.data.format_desc.blocking.inner_idxs;

  if (ndims == 1) {
    return dnnl::memory::format_tag::x;
  } else if (ndims == 2) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1]) {
        return dnnl::memory::format_tag::nc;
      } else {
        return dnnl::memory::format_tag::cn;
      }
    }
  } else if (ndims == 3) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2]) {
        return dnnl::memory::format_tag::ncw;
      } else if (strides[1] >= strides[0] && strides[0] >= strides[2]) {
        return dnnl::memory::format_tag::ntc;
      } else {
        return dnnl::memory::format_tag::nwc;
      }
    }
  } else if (ndims == 4) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3]) {
        return dnnl::memory::format_tag::nchw;
      } else if (strides[2] >= strides[3] && strides[3] >= strides[1] &&
                 strides[1] >= strides[0]) {
        return dnnl::memory::format_tag::cdba;
      } else if (strides[3] >= strides[2] && strides[2] >= strides[0] &&
                 strides[0] >= strides[1]) {
        return dnnl::memory::format_tag::dcab;
      } else {
        return dnnl::memory::format_tag::nhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw16c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw8c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return dnnl::memory::format_tag::Acdb8a;
        }
      } else if (inner_blks[0] == 4 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw4c;
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return dnnl::memory::format_tag::Acdb16a;
        }
      }
    } else if (inner_nblks == 2) {
      if (inner_blks[0] == 16 && inner_blks[1] == 16) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return dnnl::memory::format_tag::OIhw16i16o;
        }
      } else if (inner_blks[0] == 8 && inner_blks[1] == 8) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return dnnl::memory::format_tag::OIhw8i8o;
        }
      }
    }
  } else if (ndims == 5) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4]) {
        return dnnl::memory::format_tag::abcde;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[1] &&
                 strides[1] >= strides[3] && strides[3] >= strides[4]) {
        return dnnl::memory::format_tag::acbde;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
                 strides[3] >= strides[4] && strides[4] >= strides[1]) {
        return dnnl::memory::format_tag::acdeb;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return dnnl::memory::format_tag::Acdeb8a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::Abcde8a;
        }
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::aBcde8b;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return dnnl::memory::format_tag::Acdeb16a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::Abcde16a;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::aBcde16b;
        }
      }
    }
  } else if (ndims == 6) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4] &&
          strides[4] >= strides[5]) {
        return dnnl::memory::format_tag::abcdef;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[1] &&
                 strides[1] >= strides[3] && strides[3] >= strides[4] &&
                 strides[4] >= strides[5]) {
        return dnnl::memory::format_tag::acbdef;
      }
    }
  }
  // DEBUG CODE - KEEP UNTILL TENSOR.MEMORY_DESC IMPLEMENTED
  // std::cout<<"@@@@@@@@@@ UNDEFINED FORMAT @@@@@@@@@@@@@@@@@@@"<<std::endl;
  // std::cout<<"NDIMS: "<<ndims<<std::endl;
  // std::cout<<"INNER_NBLKS: "<<inner_nblks<<std::endl;
  // for (int i=0;i<ndims;++i) {
  //   std::cout<<"STRIDE["<<i<<"]: "<<strides[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_BLKS["<<i<<"]: "<<inner_blks[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_IDXS["<<i<<"]: "<<inner_idxs[i]<<std::endl;
  // }
  return dnnl::memory::format_tag::undef;
}

inline dnnl::memory::format_tag GetMKLDNNFormat(const dnnl::memory memory) {
  auto mem_desc = memory.get_desc();
  return GetMKLDNNFormat(mem_desc);
}

inline dnnl::memory::format_tag GetPlainMKLDNNFormat(int tensor_rank) {
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
      PADDLE_THROW(platform::errors::Unimplemented(
          "Paddle support tensors with rank in range <1, 9>, but received "
          "tensor with rank: %d",
          tensor_rank));
  }
}

inline MKLDNNMemoryFormat MKLDNNFormatForSize(size_t dims_size,
                                              MKLDNNMemoryFormat data_format) {
  if (dims_size == 1) {
    return MKLDNNMemoryFormat::x;
  } else if (dims_size == 2) {
    return MKLDNNMemoryFormat::nc;
  } else if (dims_size == 3) {
    if (data_format == MKLDNNMemoryFormat::nchw) {
      return MKLDNNMemoryFormat::ncw;
    } else if (data_format == MKLDNNMemoryFormat::nhwc) {
      return MKLDNNMemoryFormat::nwc;
    }
  } else if (dims_size == 4) {
    if (data_format == MKLDNNMemoryFormat::goihw) {
      return MKLDNNMemoryFormat::oihw;
    }
  } else if (dims_size == 5) {
    if (data_format == MKLDNNMemoryFormat::goidhw) {
      return MKLDNNMemoryFormat::oidhw;
    }
    if (data_format == MKLDNNMemoryFormat::nchw) {
      return MKLDNNMemoryFormat::ncdhw;
    } else if (data_format == MKLDNNMemoryFormat::nhwc) {
      return MKLDNNMemoryFormat::ndhwc;
    }
  } else if (dims_size == 6) {
    if (data_format == MKLDNNMemoryFormat::nchw) {
      return MKLDNNMemoryFormat::abcdef;
    }
  }
  return data_format;
}

inline MKLDNNMemoryFormat data_format_to_memory_format(
    const std::string& data_format) {
  switch (framework::StringToDataLayout(data_format)) {
    case framework::DataLayout::kNHWC:
      return MKLDNNMemoryFormat::nhwc;
    case framework::DataLayout::kNCHW:
      return MKLDNNMemoryFormat::nchw;
    default:
      return MKLDNNMemoryFormat::any;
  }
}

inline MKLDNNMemoryFormat StringToMKLDNNFormat(std::string* format) {
  std::transform(format->begin(), format->end(), format->begin(), ::tolower);

  if (!format->compare("nchw")) {
    return MKLDNNMemoryFormat::nchw;
  } else if (!format->compare("nchw16c")) {
    return MKLDNNMemoryFormat::nChw16c;
  } else if (!format->compare("nchw8c")) {
    return MKLDNNMemoryFormat::nChw8c;
  } else if (!format->compare("nhwc")) {
    return MKLDNNMemoryFormat::nhwc;
  } else {
    return MKLDNNMemoryFormat::any;
  }
}

inline std::string ThreadIDasStr(void) {
  return std::to_string(
      std::hash<std::thread::id>()(std::this_thread::get_id()));
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

// If MKLDNN build and CPU place then register suffix in DeviceContext
inline void AttachPointerHashToMKLDNNKey(void* ptr,
                                         const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    // Static vars will remember first executor and its thread
    // so both of them need to be processed by the same thread within
    // critical section
    static std::mutex static_vars_barrier;
    static_vars_barrier.lock();
    static auto first_exec = ptr;
    static auto first_thread = ThreadIDasStr();
    static_vars_barrier.unlock();

    if (first_exec != ptr) {
      paddle::platform::MKLDNNDeviceContext::tls().set_key_suffix(
          "E" + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    }
    // Let's register adress of current executor
    paddle::platform::MKLDNNDeviceContext::tls().set_curr_exec(ptr);

    // For first thread
    if (first_thread == ThreadIDasStr()) {
      paddle::platform::MKLDNNDeviceContext::tls().disable_tid_in_key();
    }
  }
}

template <typename... ArgTypes>
inline std::string CreateKey(const platform::MKLDNNDeviceContext& dev_ctx,
                             ArgTypes&&... args) {
  std::string key;
  key.reserve(64);
  using expand_type = int[];
  expand_type{0, (AppendKey(&key, std::forward<ArgTypes>(args)), 0)...};
  key += paddle::platform::MKLDNNDeviceContext::tls().get_key_suffix();
  return key;
}

inline std::string ExtendKeyWithThreadInfoIfNeeded(
    const platform::MKLDNNDeviceContext& dev_ctx, const std::string& key) {
  return (paddle::platform::MKLDNNDeviceContext::tls().is_tid_used_in_key() ==
          true)
             ? key + "-t:" + ThreadIDasStr()
             : key;
}

inline std::vector<std::vector<int64_t>> ToMkldnnPadding(
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

inline bool HasOpINT8DataType(const paddle::framework::OpDesc* op) {
  return (op->GetAttrIfExists<std::string>("mkldnn_data_type") == "int8" ||
          op->GetAttrIfExists<bool>("use_quantizer"));
}

inline bool HasOpBFLOAT16DataType(const paddle::framework::OpDesc* op) {
  return op->GetAttrIfExists<std::string>("mkldnn_data_type") == "bfloat16";
}

inline bool HasOpFLOAT32DataType(const paddle::framework::OpDesc* op) {
  return op->GetAttrIfExists<std::string>("mkldnn_data_type") == "float32";
}

enum class RNNReorderType { PP_NTC, PP_TNC, NTC_PP, TNC_PP };

template <typename T>
bool constexpr is_int8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}

}  // namespace platform
}  // namespace paddle
