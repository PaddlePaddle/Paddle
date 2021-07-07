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
#include "mkldnn.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
#ifdef PADDLE_WITH_MKLDNN
using MKLDNNMemoryFormat = mkldnn::memory::format_tag;
#endif
namespace platform {

using MKLDNNStream = mkldnn::stream;
using MKLDNNEngine = mkldnn::engine;
using MKLDNNMemory = mkldnn::memory;
using MKLDNNMemoryDescriptor = mkldnn::memory::desc;
using MKLDNNPrimitive = mkldnn::primitive;
using MKLDNNPrimitiveDesc = mkldnn::handle<mkldnn_primitive_desc_t>;

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
  auto desc = tf_desc<Type>(mkldnn::prop_kind::forward, (args)...);
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

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int64_t>& dims,
                                          mkldnn::memory::data_type data_type,
                                          MKLDNNMemoryFormat format) {
  return mkldnn::memory::desc({dims}, data_type, format);
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
mkldnn::memory::data_type MKLDNNGetDataType() {
  return mkldnn::memory::data_type::undef;
}

template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<float>() {
  return mkldnn::memory::data_type::f32;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<int32_t>() {
  return mkldnn::memory::data_type::s32;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<int8_t>() {
  return mkldnn::memory::data_type::s8;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<uint8_t>() {
  return mkldnn::memory::data_type::u8;
}

template <>
inline mkldnn::memory::data_type
MKLDNNGetDataType<paddle::platform::bfloat16>() {
  return mkldnn::memory::data_type::bf16;
}

inline void Reorder(mkldnn::memory src, mkldnn::memory dst,
                    const mkldnn::engine& engine) {
  auto reorder_prim = mkldnn::reorder(src, dst);
  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  platform::RecordEvent record_reorder("int_reorder",
                                       platform::EventRole::kUniqueOp);
  reorder_prim.execute(astream, src, dst);
  astream.wait();
}

inline mkldnn::memory::format_tag GetMKLDNNFormat(
    mkldnn::memory::desc mem_desc) {
  auto ndims = mem_desc.data.ndims;
  auto strides = mem_desc.data.format_desc.blocking.strides;
  auto inner_nblks = mem_desc.data.format_desc.blocking.inner_nblks;
  auto inner_blks = mem_desc.data.format_desc.blocking.inner_blks;
  auto inner_idxs = mem_desc.data.format_desc.blocking.inner_idxs;

  if (ndims == 1) {
    return mkldnn::memory::format_tag::x;
  } else if (ndims == 2) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1]) {
        return mkldnn::memory::format_tag::nc;
      } else {
        return mkldnn::memory::format_tag::cn;
      }
    }
  } else if (ndims == 3) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2]) {
        return mkldnn::memory::format_tag::ncw;
      } else if (strides[1] >= strides[0] && strides[0] >= strides[2]) {
        return mkldnn::memory::format_tag::ntc;
      } else {
        return mkldnn::memory::format_tag::nwc;
      }
    }
  } else if (ndims == 4) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3]) {
        return mkldnn::memory::format_tag::nchw;
      } else if (strides[2] >= strides[3] && strides[3] >= strides[1] &&
                 strides[1] >= strides[0]) {
        return mkldnn::memory::format_tag::cdba;
      } else {
        return mkldnn::memory::format_tag::nhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw16c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw8c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdb8a;
        }
      } else if (inner_blks[0] == 4 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw4c;
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdb16a;
        }
      }
    } else if (inner_nblks == 2) {
      if (inner_blks[0] == 16 && inner_blks[1] == 16) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return mkldnn::memory::format_tag::OIhw16i16o;
        }
      } else if (inner_blks[0] == 8 && inner_blks[1] == 8) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return mkldnn::memory::format_tag::OIhw8i8o;
        }
      }
    }
  } else if (ndims == 5) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4]) {
        return mkldnn::memory::format_tag::ncdhw;
      } else {
        return mkldnn::memory::format_tag::ndhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdeb8a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::Abcde8a;
        }
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::aBcde8b;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdeb16a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::Abcde16a;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::aBcde16b;
        }
      }
    }
  } else if (ndims == 6) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4] &&
          strides[4] >= strides[5]) {
        return mkldnn::memory::format_tag::abcdef;
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
  return mkldnn::memory::format_tag::undef;
}

inline mkldnn::memory::format_tag GetMKLDNNFormat(const mkldnn::memory memory) {
  auto mem_desc = memory.get_desc();
  return GetMKLDNNFormat(mem_desc);
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
                      const mkldnn::memory::format_tag& format) {
  key->append(std::to_string(static_cast<int>(format)));
}

template <>
inline void AppendKey(std::string* key,
                      const mkldnn::memory::data_type& data_type) {
  key->append(std::to_string(static_cast<int>(data_type)));
}

template <>
inline void AppendKey(std::string* key, const mkldnn::algorithm& algorithm) {
  key->append(std::to_string(static_cast<int>(algorithm)));
}

template <>
inline void AppendKey(std::string* key,
                      const mkldnn::normalization_flags& flags) {
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
  return ((paddle::platform::MKLDNNDeviceContext::tls().is_tid_used_in_key() ==
           true) &&
          (platform::MKLDNNDeviceContext::tls().get_cur_mkldnn_session_id() ==
           platform::MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_Default))
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

inline std::string MKLDNNFormatToString(MKLDNNMemoryFormat format_tag) {
  switch (format_tag) {
    case MKLDNNMemoryFormat::undef:
      return "undef";
    case MKLDNNMemoryFormat::any:
      return "any";
    case MKLDNNMemoryFormat::a:
      return "a";
    case MKLDNNMemoryFormat::ab:
      return "ab";
    case MKLDNNMemoryFormat::ba:
      return "ba";
    case MKLDNNMemoryFormat::abc:
      return "abc";
    case MKLDNNMemoryFormat::acb:
      return "acb";
    case MKLDNNMemoryFormat::bac:
      return "bac";
    case MKLDNNMemoryFormat::bca:
      return "bca";
    case MKLDNNMemoryFormat::cba:
      return "cba";
    case MKLDNNMemoryFormat::abcd:
      return "abcd";
    case MKLDNNMemoryFormat::abdc:
      return "abdc";
    case MKLDNNMemoryFormat::acdb:
      return "acdb";
    case MKLDNNMemoryFormat::bacd:
      return "bacd";
    case MKLDNNMemoryFormat::bcda:
      return "bcda";
    case MKLDNNMemoryFormat::cdba:
      return "cdba";
    case MKLDNNMemoryFormat::dcab:
      return "dcab";
    case MKLDNNMemoryFormat::abcde:
      return "abcde";
    case MKLDNNMemoryFormat::abdec:
      return "abdec";
    case MKLDNNMemoryFormat::acbde:
      return "acbde";
    case MKLDNNMemoryFormat::acdeb:
      return "acdeb";
    case MKLDNNMemoryFormat::bacde:
      return "bacde";
    case MKLDNNMemoryFormat::bcdea:
      return "bcdea";
    case MKLDNNMemoryFormat::cdeba:
      return "cdeba";
    case MKLDNNMemoryFormat::decab:
      return "decab";
    case MKLDNNMemoryFormat::abced:
      return "abced";
    case MKLDNNMemoryFormat::abcdef:
      return "abcdef";
    case MKLDNNMemoryFormat::abdfce:
      return "abdfce";
    case MKLDNNMemoryFormat::acbdef:
      return "acbdef";
    case MKLDNNMemoryFormat::abdefc:
      return "abdefc";
    case MKLDNNMemoryFormat::defcab:
      return "defcab";
    case MKLDNNMemoryFormat::abcdfe:
      return "abcdfe";
    case MKLDNNMemoryFormat::abcdefg:
      return "abcdefg";
    case MKLDNNMemoryFormat::abcdegf:
      return "abcdegf";
    case MKLDNNMemoryFormat::abcdefgh:
      return "abcdefgh";
    case MKLDNNMemoryFormat::abcdefhg:
      return "abcdefhg";
    case MKLDNNMemoryFormat::abcdefghi:
      return "abcdefghi";
    case MKLDNNMemoryFormat::abcdefgih:
      return "abcdefgih";
    case MKLDNNMemoryFormat::abcdefghij:
      return "abcdefghij";
    case MKLDNNMemoryFormat::abcdefghji:
      return "abcdefghji";
    case MKLDNNMemoryFormat::abcdefghijk:
      return "abcdefghijk";
    case MKLDNNMemoryFormat::abcdefghikj:
      return "abcdefghikj";
    case MKLDNNMemoryFormat::abcdefghijkl:
      return "abcdefghijkl";
    case MKLDNNMemoryFormat::abcdefghijlk:
      return "abcdefghijlk";
    case MKLDNNMemoryFormat::AB16b16a:
      return "AB16b16a";
    case MKLDNNMemoryFormat::AB16b32a:
      return "AB16b32a";
    case MKLDNNMemoryFormat::AB16b64a:
      return "AB16b64a";
    case MKLDNNMemoryFormat::AB8b16a2b:
      return "AB8b16a2b";
    case MKLDNNMemoryFormat::AB8b32a2b:
      return "AB8b32a2b";
    case MKLDNNMemoryFormat::AB8b64a2b:
      return "AB8b64a2b";
    case MKLDNNMemoryFormat::AB4b16a4b:
      return "AB4b16a4b";
    case MKLDNNMemoryFormat::AB4b32a4b:
      return "AB4b32a4b";
    case MKLDNNMemoryFormat::AB4b64a4b:
      return "AB4b64a4b";
    case MKLDNNMemoryFormat::AB16b16a4b:
      return "AB16b16a4b";
    case MKLDNNMemoryFormat::AB16b32a4b:
      return "AB16b32a4b";
    case MKLDNNMemoryFormat::AB16b48a4b:
      return "AB16b48a4b";
    case MKLDNNMemoryFormat::AB16b64a4b:
      return "AB16b64a4b";
    case MKLDNNMemoryFormat::AB16b16a2b:
      return "AB16b16a2b";
    case MKLDNNMemoryFormat::AB16b32a2b:
      return "AB16b32a2b";
    case MKLDNNMemoryFormat::AB16b48a2b:
      return "AB16b48a2b";
    case MKLDNNMemoryFormat::AB16b64a2b:
      return "AB16b64a2b";
    case MKLDNNMemoryFormat::Abc16a:
      return "Abc16a";
    case MKLDNNMemoryFormat::ABc16a16b:
      return "ABc16a16b";
    case MKLDNNMemoryFormat::ABc4a4b:
      return "ABc4a4b";
    case MKLDNNMemoryFormat::aBc16b:
      return "aBc16b";
    case MKLDNNMemoryFormat::aBc32b:
      return "aBc32b";
    case MKLDNNMemoryFormat::ABc16b16a:
      return "ABc16b16a";
    case MKLDNNMemoryFormat::ABc16b32a:
      return "ABc16b32a";
    case MKLDNNMemoryFormat::ABc16b64a:
      return "ABc16b64a";
    case MKLDNNMemoryFormat::Abc4a:
      return "Abc4a";
    case MKLDNNMemoryFormat::aBc4b:
      return "aBc4b";
    case MKLDNNMemoryFormat::ABc4b16a4b:
      return "ABc4b16a4b";
    case MKLDNNMemoryFormat::ABc4b32a4b:
      return "ABc4b32a4b";
    case MKLDNNMemoryFormat::ABc4b64a4b:
      return "ABc4b64a4b";
    case MKLDNNMemoryFormat::ABc2b8a4b:
      return "ABc2b8a4b";
    case MKLDNNMemoryFormat::ABc16a16b2a:
      return "ABc16a16b2a";
    case MKLDNNMemoryFormat::ABc16b16a4b:
      return "ABc16b16a4b";
    case MKLDNNMemoryFormat::ABc16b32a4b:
      return "ABc16b32a4b";
    case MKLDNNMemoryFormat::ABc16b48a4b:
      return "ABc16b48a4b";
    case MKLDNNMemoryFormat::ABc16b64a4b:
      return "ABc16b64a4b";
    case MKLDNNMemoryFormat::ABc16b16a2b:
      return "ABc16b16a2b";
    case MKLDNNMemoryFormat::ABc16b32a2b:
      return "ABc16b32a2b";
    case MKLDNNMemoryFormat::ABc16b48a2b:
      return "ABc16b48a2b";
    case MKLDNNMemoryFormat::ABc16b64a2b:
      return "ABc16b64a2b";
    case MKLDNNMemoryFormat::ABc4b4a:
      return "ABc4b4a";
    case MKLDNNMemoryFormat::ABc8a16b2a:
      return "ABc8a16b2a";
    case MKLDNNMemoryFormat::ABc8a8b:
      return "ABc8a8b";
    case MKLDNNMemoryFormat::ABc8a4b:
      return "ABc8a4b";
    case MKLDNNMemoryFormat::aBc8b:
      return "aBc8b";
    case MKLDNNMemoryFormat::ABc8b16a2b:
      return "ABc8b16a2b";
    case MKLDNNMemoryFormat::ABc8b32a2b:
      return "ABc8b32a2b";
    case MKLDNNMemoryFormat::ABc8b64a2b:
      return "ABc8b64a2b";
    case MKLDNNMemoryFormat::ABc8b8a:
      return "ABc8b8a";
    case MKLDNNMemoryFormat::Abcd8a:
      return "Abcd8a";
    case MKLDNNMemoryFormat::Abcd16a:
      return "Abcd16a";
    case MKLDNNMemoryFormat::Abcd32a:
      return "Abcd32a";
    case MKLDNNMemoryFormat::ABcd16a16b:
      return "ABcd16a16b";
    case MKLDNNMemoryFormat::aBcd16b:
      return "aBcd16b";
    case MKLDNNMemoryFormat::aBcd32b:
      return "aBcd32b";
    case MKLDNNMemoryFormat::ABcd16b16a:
      return "ABcd16b16a";
    case MKLDNNMemoryFormat::ABcd16b32a:
      return "ABcd16b32a";
    case MKLDNNMemoryFormat::ABcd16b64a:
      return "ABcd16b64a";
    case MKLDNNMemoryFormat::aBCd16b16c:
      return "aBCd16b16c";
    case MKLDNNMemoryFormat::aBCd16c16b:
      return "aBCd16c16b";
    case MKLDNNMemoryFormat::Abcd4a:
      return "Abcd4a";
    case MKLDNNMemoryFormat::aBcd4b:
      return "aBcd4b";
    case MKLDNNMemoryFormat::ABcd4b16a4b:
      return "ABcd4b16a4b";
    case MKLDNNMemoryFormat::ABcd4b32a4b:
      return "ABcd4b32a4b";
    case MKLDNNMemoryFormat::ABcd4b64a4b:
      return "ABcd4b64a4b";
    case MKLDNNMemoryFormat::ABcd2b8a4b:
      return "ABcd2b8a4b";
    case MKLDNNMemoryFormat::ABcd4b4a:
      return "ABcd4b4a";
    case MKLDNNMemoryFormat::ABcd4a4b:
      return "ABcd4a4b";
    case MKLDNNMemoryFormat::aBCd4c16b4c:
      return "aBCd4c16b4c";
    case MKLDNNMemoryFormat::aBCd2c8b4c:
      return "aBCd2c8b4c";
    case MKLDNNMemoryFormat::ABcd16a16b2a:
      return "ABcd16a16b2a";
    case MKLDNNMemoryFormat::ABcd16b16a4b:
      return "ABcd16b16a4b";
    case MKLDNNMemoryFormat::ABcd16b32a4b:
      return "ABcd16b32a4b";
    case MKLDNNMemoryFormat::ABcd16b48a4b:
      return "ABcd16b48a4b";
    case MKLDNNMemoryFormat::ABcd16b64a4b:
      return "ABcd16b64a4b";
    case MKLDNNMemoryFormat::ABcd16b16a2b:
      return "ABcd16b16a2b";
    case MKLDNNMemoryFormat::ABcd16b32a2b:
      return "ABcd16b32a2b";
    case MKLDNNMemoryFormat::ABcd16b48a2b:
      return "ABcd16b48a2b";
    case MKLDNNMemoryFormat::ABcd16b64a2b:
      return "ABcd16b64a2b";
    case MKLDNNMemoryFormat::aBCd16b16c2b:
      return "aBCd16b16c2b";
    case MKLDNNMemoryFormat::aBCd16c16b4c:
      return "aBCd16c16b4c";
    case MKLDNNMemoryFormat::aBCd16c16b2c:
      return "aBCd16c16b2c";
    case MKLDNNMemoryFormat::aBCd4c4b:
      return "aBCd4c4b";
    case MKLDNNMemoryFormat::aBCd4b4c:
      return "aBCd4b4c";
    case MKLDNNMemoryFormat::ABcd8a16b2a:
      return "ABcd8a16b2a";
    case MKLDNNMemoryFormat::ABcd8a8b:
      return "ABcd8a8b";
    case MKLDNNMemoryFormat::ABcd8a4b:
      return "ABcd8a4b";
    case MKLDNNMemoryFormat::aBcd8b:
      return "aBcd8b";
    case MKLDNNMemoryFormat::ABcd8b16a2b:
      return "ABcd8b16a2b";
    case MKLDNNMemoryFormat::ABcd8b32a2b:
      return "ABcd8b32a2b";
    case MKLDNNMemoryFormat::ABcd8b64a2b:
      return "ABcd8b64a2b";
    case MKLDNNMemoryFormat::aBCd8b16c2b:
      return "aBCd8b16c2b";
    case MKLDNNMemoryFormat::ABcd8b8a:
      return "ABcd8b8a";
    case MKLDNNMemoryFormat::aBCd8b8c:
      return "aBCd8b8c";
    case MKLDNNMemoryFormat::aBCd8b4c:
      return "aBCd8b4c";
    case MKLDNNMemoryFormat::aBCd8c16b2c:
      return "aBCd8c16b2c";
    case MKLDNNMemoryFormat::aBCd8c8b:
      return "aBCd8c8b";
    case MKLDNNMemoryFormat::Abcde16a:
      return "Abcde16a";
    case MKLDNNMemoryFormat::Abcde32a:
      return "Abcde32a";
    case MKLDNNMemoryFormat::ABcde16a16b:
      return "ABcde16a16b";
    case MKLDNNMemoryFormat::aBcde16b:
      return "aBcde16b";
    case MKLDNNMemoryFormat::aBcde32b:
      return "aBcde32b";
    case MKLDNNMemoryFormat::ABcde16b16a:
      return "ABcde16b16a";
    case MKLDNNMemoryFormat::ABcde16b32a:
      return "ABcde16b32a";
    case MKLDNNMemoryFormat::ABcde16b64a:
      return "ABcde16b64a";
    case MKLDNNMemoryFormat::aBCde16b16c:
      return "aBCde16b16c";
    case MKLDNNMemoryFormat::aBCde16c16b:
      return "aBCde16c16b";
    case MKLDNNMemoryFormat::aBCde2c8b4c:
      return "aBCde2c8b4c";
    case MKLDNNMemoryFormat::Abcde4a:
      return "Abcde4a";
    case MKLDNNMemoryFormat::aBcde4b:
      return "aBcde4b";
    case MKLDNNMemoryFormat::ABcde4b4a:
      return "ABcde4b4a";
    case MKLDNNMemoryFormat::ABcde4a4b:
      return "ABcde4a4b";
    case MKLDNNMemoryFormat::aBCde4b4c:
      return "aBCde4b4c";
    case MKLDNNMemoryFormat::aBCde4c16b4c:
      return "aBCde4c16b4c";
    case MKLDNNMemoryFormat::aBCde16b16c2b:
      return "aBCde16b16c2b";
    case MKLDNNMemoryFormat::aBCde16c16b4c:
      return "aBCde16c16b4c";
    case MKLDNNMemoryFormat::aBCde16c16b2c:
      return "aBCde16c16b2c";
    case MKLDNNMemoryFormat::aBCdef16c16b2c:
      return "aBCdef16c16b2c";
    case MKLDNNMemoryFormat::aBCde4c4b:
      return "aBCde4c4b";
    case MKLDNNMemoryFormat::Abcde8a:
      return "Abcde8a";
    case MKLDNNMemoryFormat::ABcde8a8b:
      return "ABcde8a8b";
    case MKLDNNMemoryFormat::ABcde8a4b:
      return "ABcde8a4b";
    case MKLDNNMemoryFormat::aBcde8b:
      return "aBcde8b";
    case MKLDNNMemoryFormat::ABcde8b16a2b:
      return "ABcde8b16a2b";
    case MKLDNNMemoryFormat::ABcde8b32a2b:
      return "ABcde8b32a2b";
    case MKLDNNMemoryFormat::ABcde8b64a2b:
      return "ABcde8b64a2b";
    case MKLDNNMemoryFormat::ABcde4b16a4b:
      return "ABcde4b16a4b";
    case MKLDNNMemoryFormat::ABcde4b32a4b:
      return "ABcde4b32a4b";
    case MKLDNNMemoryFormat::ABcde4b64a4b:
      return "ABcde4b64a4b";
    case MKLDNNMemoryFormat::ABcde16b16a4b:
      return "ABcde16b16a4b";
    case MKLDNNMemoryFormat::ABcde16b32a4b:
      return "ABcde16b32a4b";
    case MKLDNNMemoryFormat::ABcde16b48a4b:
      return "ABcde16b48a4b";
    case MKLDNNMemoryFormat::ABcde16b64a4b:
      return "ABcde16b64a4b";
    case MKLDNNMemoryFormat::ABcde16b16a2b:
      return "ABcde16b16a2b";
    case MKLDNNMemoryFormat::ABcde16b32a2b:
      return "ABcde16b32a2b";
    case MKLDNNMemoryFormat::ABcde16b48a2b:
      return "ABcde16b48a2b";
    case MKLDNNMemoryFormat::ABcde16b64a2b:
      return "ABcde16b64a2b";
    case MKLDNNMemoryFormat::ABcde2b8a4b:
      return "ABcde2b8a4b";
    case MKLDNNMemoryFormat::aBCde8b16c2b:
      return "aBCde8b16c2b";
    case MKLDNNMemoryFormat::ABcde8b8a:
      return "ABcde8b8a";
    case MKLDNNMemoryFormat::aBCde8b8c:
      return "aBCde8b8c";
    case MKLDNNMemoryFormat::aBCde8b4c:
      return "aBCde8b4c";
    case MKLDNNMemoryFormat::ABcd4a8b8a4b:
      return "ABcd4a8b8a4b";
    case MKLDNNMemoryFormat::ABcd2a8b8a2b:
      return "ABcd2a8b8a2b";
    case MKLDNNMemoryFormat::aBCde4b8c8b4c:
      return "aBCde4b8c8b4c";
    case MKLDNNMemoryFormat::aBCde2b8c8b2c:
      return "aBCde2b8c8b2c";
    case MKLDNNMemoryFormat::aBCde8c16b2c:
      return "aBCde8c16b2c";
    case MKLDNNMemoryFormat::aBCde8c8b:
      return "aBCde8c8b";
    case MKLDNNMemoryFormat::aBcdef16b:
      return "aBcdef16b";
    case MKLDNNMemoryFormat::aBCdef16b16c:
      return "aBCdef16b16c";
    case MKLDNNMemoryFormat::aBCdef16c16b:
      return "aBCdef16c16b";
    case MKLDNNMemoryFormat::aBcdef4b:
      return "aBcdef4b";
    case MKLDNNMemoryFormat::aBCdef2c8b4c:
      return "aBCdef2c8b4c";
    case MKLDNNMemoryFormat::aBCdef4c4b:
      return "aBCdef4c4b";
    case MKLDNNMemoryFormat::aBCdef4b4c:
      return "aBCdef4b4c";
    case MKLDNNMemoryFormat::aBCdef8b8c:
      return "aBCdef8b8c";
    case MKLDNNMemoryFormat::aBCdef8b4c:
      return "aBCdef8b4c";
    case MKLDNNMemoryFormat::aBCdef8c16b2c:
      return "aBCdef8c16b2c";
    case MKLDNNMemoryFormat::aBCdef4c16b4c:
      return "aBCdef4c16b4c";
    case MKLDNNMemoryFormat::aBCdef8c8b:
      return "aBCdef8c8b";
    case MKLDNNMemoryFormat::aBdc16b:
      return "aBdc16b";
    case MKLDNNMemoryFormat::aBdc4b:
      return "aBdc4b";
    case MKLDNNMemoryFormat::aBdc8b:
      return "aBdc8b";
    case MKLDNNMemoryFormat::aBdec16b:
      return "aBdec16b";
    case MKLDNNMemoryFormat::aBdec4b:
      return "aBdec4b";
    case MKLDNNMemoryFormat::aBdec8b:
      return "aBdec8b";
    case MKLDNNMemoryFormat::aBdefc16b:
      return "aBdefc16b";
    case MKLDNNMemoryFormat::aCBdef16c16b:
      return "aCBdef16c16b";
    case MKLDNNMemoryFormat::aCBdef16b16c:
      return "aCBdef16b16c";
    case MKLDNNMemoryFormat::aBdefc4b:
      return "aBdefc4b";
    case MKLDNNMemoryFormat::aBdefc8b:
      return "aBdefc8b";
    case MKLDNNMemoryFormat::Acb16a:
      return "Acb16a";
    case MKLDNNMemoryFormat::Acb4a:
      return "Acb4a";
    case MKLDNNMemoryFormat::Acb8a:
      return "Acb8a";
    case MKLDNNMemoryFormat::aCBd16b16c:
      return "aCBd16b16c";
    case MKLDNNMemoryFormat::aCBd16c16b:
      return "aCBd16c16b";
    case MKLDNNMemoryFormat::aCBde16b16c:
      return "aCBde16b16c";
    case MKLDNNMemoryFormat::aCBde16c16b:
      return "aCBde16c16b";
    case MKLDNNMemoryFormat::Acdb16a:
      return "Acdb16a";
    case MKLDNNMemoryFormat::Acdb4a:
      return "Acdb4a";
    case MKLDNNMemoryFormat::Acdb8a:
      return "Acdb8a";
    case MKLDNNMemoryFormat::Acdeb16a:
      return "Acdeb16a";
    case MKLDNNMemoryFormat::Acdeb4a:
      return "Acdeb4a";
    case MKLDNNMemoryFormat::Acdeb8a:
      return "Acdeb8a";
    case MKLDNNMemoryFormat::BAc16a16b:
      return "BAc16a16b";
    case MKLDNNMemoryFormat::BAc16b16a:
      return "BAc16b16a";
    case MKLDNNMemoryFormat::BAcd16a16b:
      return "BAcd16a16b";
    case MKLDNNMemoryFormat::BAcd16b16a:
      return "BAcd16b16a";
    case MKLDNNMemoryFormat::ABcd32a32b:
      return "ABcd32a32b";
    case MKLDNNMemoryFormat::BAcde16b16a:
      return "BAcde16b16a";
    case MKLDNNMemoryFormat::BAcde16a16b:
      return "BAcde16a16b";
    case MKLDNNMemoryFormat::aBdec32b:
      return "aBdec32b";
    case MKLDNNMemoryFormat::Abcdef16a:
      return "Abcdef16a";
    case MKLDNNMemoryFormat::Abcdef32a:
      return "Abcdef32a";
    case MKLDNNMemoryFormat::Acdb32a:
      return "Acdb32a";
    case MKLDNNMemoryFormat::aBCd2b4c2b:
      return "aBCd2b4c2b";
    case MKLDNNMemoryFormat::aBCde2b4c2b:
      return "aBCde2b4c2b";
    case MKLDNNMemoryFormat::aBCdef2b4c2b:
      return "aBCdef2b4c2b";
    case MKLDNNMemoryFormat::aBCd2c4b2c:
      return "aBCd2c4b2c";
    case MKLDNNMemoryFormat::aBCde2c4b2c:
      return "aBCde2c4b2c";
    case MKLDNNMemoryFormat::aBCdef2c4b2c:
      return "aBCdef2c4b2c";
    case MKLDNNMemoryFormat::aBCd4b8c2b:
      return "aBCd4b8c2b";
    case MKLDNNMemoryFormat::aBCde4b8c2b:
      return "aBCde4b8c2b";
    case MKLDNNMemoryFormat::aBCdef4b8c2b:
      return "aBCdef4b8c2b";
    case MKLDNNMemoryFormat::aBCd4c8b2c:
      return "aBCd4c8b2c";
    case MKLDNNMemoryFormat::aBCde4c8b2c:
      return "aBCde4c8b2c";
    case MKLDNNMemoryFormat::aBCdef4c8b2c:
      return "aBCdef4c8b2c";
    case MKLDNNMemoryFormat::AB32a32b8a4b:
      return "AB32a32b8a4b";
    case MKLDNNMemoryFormat::AB32a32b8a2b:
      return "AB32a32b8a2b";
    case MKLDNNMemoryFormat::AB8a4b:
      return "AB8a4b";
    case MKLDNNMemoryFormat::AB8a2b:
      return "AB8a2b";
    case MKLDNNMemoryFormat::abDc32d:
      return "abDc32d";
    case MKLDNNMemoryFormat::abDC32d4c:
      return "abDC32d4c";
    case MKLDNNMemoryFormat::abCd32c:
      return "abCd32c";
    case MKLDNNMemoryFormat::abdEc32e:
      return "abdEc32e";
    case MKLDNNMemoryFormat::abdEC32e2c:
      return "abdEC32e2c";
    case MKLDNNMemoryFormat::abdEC32e4c:
      return "abdEC32e4c";
    case MKLDNNMemoryFormat::abdCe32c:
      return "abdCe32c";
    case MKLDNNMemoryFormat::abdCE32c2e:
      return "abdCE32c2e";
    case MKLDNNMemoryFormat::aBCdef16c16b4c:
      return "aBCdef16c16b4c";
    case MKLDNNMemoryFormat::aBdC16b4c:
      return "aBdC16b4c";
    case MKLDNNMemoryFormat::aBdeC16b4c:
      return "aBdeC16b4c";
    case MKLDNNMemoryFormat::AcB16a4b:
      return "AcB16a4b";
    case MKLDNNMemoryFormat::AcdB16a2b:
      return "AcdB16a2b";
    case MKLDNNMemoryFormat::aBdefC16b4c:
      return "aBdefC16b4c";
    case MKLDNNMemoryFormat::AcdeB16a4b:
      return "AcdeB16a4b";
    case MKLDNNMemoryFormat::Acb32a:
      return "Acb32a";
    case MKLDNNMemoryFormat::AcB32a2b:
      return "AcB32a2b";
    case MKLDNNMemoryFormat::AcB32a4b:
      return "AcB32a4b";
    case MKLDNNMemoryFormat::Acb48a:
      return "Acb48a";
    case MKLDNNMemoryFormat::AcB48a2b:
      return "AcB48a2b";
    case MKLDNNMemoryFormat::AcB48a4b:
      return "AcB48a4b";
    case MKLDNNMemoryFormat::Acb64a:
      return "Acb64a";
    case MKLDNNMemoryFormat::AcB64a2b:
      return "AcB64a2b";
    case MKLDNNMemoryFormat::AcB64a4b:
      return "AcB64a4b";
    case MKLDNNMemoryFormat::cBa2b:
      return "cBa2b";
    case MKLDNNMemoryFormat::cBa4b:
      return "cBa4b";
    case MKLDNNMemoryFormat::aBdc32b:
      return "aBdc32b";
    case MKLDNNMemoryFormat::aBdC32b2c:
      return "aBdC32b2c";
    case MKLDNNMemoryFormat::aBdC32b4c:
      return "aBdC32b4c";
    case MKLDNNMemoryFormat::aBdc48b:
      return "aBdc48b";
    case MKLDNNMemoryFormat::aBdC48b2c:
      return "aBdC48b2c";
    case MKLDNNMemoryFormat::aBdC48b4c:
      return "aBdC48b4c";
    case MKLDNNMemoryFormat::aBdc64b:
      return "aBdc64b";
    case MKLDNNMemoryFormat::aBdC64b2c:
      return "aBdC64b2c";
    case MKLDNNMemoryFormat::aBdC64b4c:
      return "aBdC64b4c";
    case MKLDNNMemoryFormat::adcb:
      return "adcb";
    case MKLDNNMemoryFormat::adCb2c:
      return "adCb2c";
    case MKLDNNMemoryFormat::adCb4c:
      return "adCb4c";
    case MKLDNNMemoryFormat::AcdB32a2b:
      return "AcdB32a2b";
    case MKLDNNMemoryFormat::AcdB32a4b:
      return "AcdB32a4b";
    case MKLDNNMemoryFormat::Acdb48a:
      return "Acdb48a";
    case MKLDNNMemoryFormat::AcdB48a2b:
      return "AcdB48a2b";
    case MKLDNNMemoryFormat::AcdB48a4b:
      return "AcdB48a4b";
    case MKLDNNMemoryFormat::Acdb64a:
      return "Acdb64a";
    case MKLDNNMemoryFormat::AcdB64a2b:
      return "AcdB64a2b";
    case MKLDNNMemoryFormat::AcdB64a4b:
      return "AcdB64a4b";
    case MKLDNNMemoryFormat::cdBa2b:
      return "cdBa2b";
    case MKLDNNMemoryFormat::cdBa4b:
      return "cdBa4b";
    case MKLDNNMemoryFormat::aBdeC32b2c:
      return "aBdeC32b2c";
    case MKLDNNMemoryFormat::aBdeC32b4c:
      return "aBdeC32b4c";
    case MKLDNNMemoryFormat::aBdec48b:
      return "aBdec48b";
    case MKLDNNMemoryFormat::aBdeC48b2c:
      return "aBdeC48b2c";
    case MKLDNNMemoryFormat::aBdeC48b4c:
      return "aBdeC48b4c";
    case MKLDNNMemoryFormat::aBdec64b:
      return "aBdec64b";
    case MKLDNNMemoryFormat::aBdeC64b2c:
      return "aBdeC64b2c";
    case MKLDNNMemoryFormat::aBdeC64b4c:
      return "aBdeC64b4c";
    case MKLDNNMemoryFormat::adecb:
      return "adecb";
    case MKLDNNMemoryFormat::adeCb2c:
      return "adeCb2c";
    case MKLDNNMemoryFormat::adeCb4c:
      return "adeCb4c";
    case MKLDNNMemoryFormat::Acdeb32a:
      return "Acdeb32a";
    case MKLDNNMemoryFormat::AcdeB32a2b:
      return "AcdeB32a2b";
    case MKLDNNMemoryFormat::AcdeB32a4b:
      return "AcdeB32a4b";
    case MKLDNNMemoryFormat::Acdeb48a:
      return "Acdeb48a";
    case MKLDNNMemoryFormat::AcdeB48a2b:
      return "AcdeB48a2b";
    case MKLDNNMemoryFormat::AcdeB48a4b:
      return "AcdeB48a4b";
    case MKLDNNMemoryFormat::Acdeb64a:
      return "Acdeb64a";
    case MKLDNNMemoryFormat::AcdeB64a2b:
      return "AcdeB64a2b";
    case MKLDNNMemoryFormat::AcdeB64a4b:
      return "AcdeB64a4b";
    case MKLDNNMemoryFormat::cdeBa2b:
      return "cdeBa2b";
    case MKLDNNMemoryFormat::cdeBa4b:
      return "cdeBa4b";
    case MKLDNNMemoryFormat::aBdefc32b:
      return "aBdefc32b";
    case MKLDNNMemoryFormat::aBdefC32b2c:
      return "aBdefC32b2c";
    case MKLDNNMemoryFormat::aBdefC32b4c:
      return "aBdefC32b4c";
    case MKLDNNMemoryFormat::aBdefc48b:
      return "aBdefc48b";
    case MKLDNNMemoryFormat::aBdefC48b2c:
      return "aBdefC48b2c";
    case MKLDNNMemoryFormat::aBdefC48b4c:
      return "aBdefC48b4c";
    case MKLDNNMemoryFormat::aBdefc64b:
      return "aBdefc64b";
    case MKLDNNMemoryFormat::aBdefC64b2c:
      return "aBdefC64b2c";
    case MKLDNNMemoryFormat::aBdefC64b4c:
      return "aBdefC64b4c";
    case MKLDNNMemoryFormat::adefcb:
      return "adefcb";
    case MKLDNNMemoryFormat::adefCb2c:
      return "adefCb2c";
    case MKLDNNMemoryFormat::adefCb4c:
      return "adefCb4c";
    case MKLDNNMemoryFormat::format_tag_last:
      return "format_tag_last";
    case MKLDNNMemoryFormat::NCdhw32n32c:
      return "NCdhw32n32c";
    case MKLDNNMemoryFormat::OwI16o2i:
      return "OwI16o2i";
    case MKLDNNMemoryFormat::OdhwI16o2i:
      return "OdhwI16o2i";
    case MKLDNNMemoryFormat::gOwI16o2i:
      return "gOwI16o2i";
    case MKLDNNMemoryFormat::gOhwI16o2i:
      return "gOhwI16o2i";
    case MKLDNNMemoryFormat::OIw4o8i8o4i:
      return "OIw4o8i8o4i";
    case MKLDNNMemoryFormat::OIdhw4o8i8o4i:
      return "OIdhw4o8i8o4i";
    case MKLDNNMemoryFormat::gOIw4o8i8o4i:
      return "gOIw4o8i8o4i";
    case MKLDNNMemoryFormat::gOIdhw4o8i8o4i:
      return "gOIdhw4o8i8o4i";
    case MKLDNNMemoryFormat::gOdhwI16o2i:
      return "gOdhwI16o2i";
    case MKLDNNMemoryFormat::OhwI16o4i:
      return "OhwI16o4i";
  }
  return "unknown";
}

}  // namespace platform
}  // namespace paddle
