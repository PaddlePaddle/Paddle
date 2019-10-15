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

#include <mkldnn.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
#ifdef PADDLE_WITH_MKLDNN
using MKLDNNMemoryFormat = mkldnn::memory::format;
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

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int>& dims,
                                          mkldnn::memory::data_type data_type,
                                          MKLDNNMemoryFormat format) {
  mkldnn::memory::dims tz = dims;
  return mkldnn::memory::desc({tz}, data_type, format);
}

inline bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_mkldnn = ctx.Attr<bool>("use_mkldnn");
  return use_mkldnn && platform::is_cpu_place(ctx.GetPlace());
}

template <typename Type>
mkldnn::memory::data_type MKLDNNGetDataType() {
  return mkldnn::memory::data_type::data_undef;
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

inline void Reorder(const mkldnn::memory& src, const mkldnn::memory& dst) {
  auto reorder_prim = mkldnn::reorder(src, dst);
  std::vector<mkldnn::primitive> pipeline;
  pipeline.push_back(reorder_prim);
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

inline MKLDNNMemoryFormat GetMKLDNNFormat(const mkldnn::memory memory) {
  return static_cast<MKLDNNMemoryFormat>(
      memory.get_primitive_desc().desc().data.format);
}

inline MKLDNNMemoryFormat GetMKLDNNFormat(
    const mkldnn::sum::primitive_desc& memory) {
  return static_cast<MKLDNNMemoryFormat>(
      memory.dst_primitive_desc().desc().data.format);
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

inline void AppendKey(std::string* key, const std::string& str) {
  key->append(str);
}

inline void AppendKey(std::string* key, const char* str) { key->append(str); }

inline void AppendKey(std::string* key, const std::vector<int>& dims) {
  for (size_t i = 0; i < dims.size(); i++) {
    AppendKey(key, std::to_string(dims[i]));
  }
}

template <typename... ArgTypes>
inline std::string CreateKey(ArgTypes&&... args) {
  std::string key;
  key.reserve(256);
  using expand_type = int[];
  expand_type{0, (AppendKey(&key, std::forward<ArgTypes>(args)), 0)...};
  return key;
}

}  // namespace platform
}  // namespace paddle
