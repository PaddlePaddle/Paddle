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
#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
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
                                          mkldnn::memory::format format) {
  mkldnn::memory::dims tz = dims;
  return mkldnn::memory::desc({tz}, data_type, format);
}

inline bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_mkldnn = ctx.Attr<bool>("use_mkldnn");
  return use_mkldnn && platform::is_cpu_place(ctx.GetPlace());
}

template <typename Type>
mkldnn::memory::data_type MKLDNNGetDataType() {
  return mkldnn::memory::data_undef;
}

template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<float>() {
  return mkldnn::memory::f32;
}

inline void Reorder(const mkldnn::memory& src, const mkldnn::memory& dst) {
  auto reorder_prim = mkldnn::reorder(src, dst);
  std::vector<mkldnn::primitive> pipeline;
  pipeline.push_back(reorder_prim);
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

inline mkldnn::memory::format GetMKLDNNFormat(const mkldnn::memory memory) {
  return static_cast<mkldnn::memory::format>(
      memory.get_primitive_desc().desc().data.format);
}

inline mkldnn::memory::format GetMKLDNNFormat(
    const mkldnn::sum::primitive_desc& memory) {
  return static_cast<mkldnn::memory::format>(
      memory.dst_primitive_desc().desc().data.format);
}

inline mkldnn::memory::format MKLDNNFormatForSize(
    size_t dims_size, mkldnn::memory::format data_format) {
  if (dims_size == 1) {
    return mkldnn::memory::format::x;
  } else if (dims_size == 2) {
    return mkldnn::memory::format::nc;
  } else if (dims_size == 3) {
    if (data_format == mkldnn::memory::format::nchw) {
      return mkldnn::memory::format::ncw;
    } else if (data_format == mkldnn::memory::format::nhwc) {
      return mkldnn::memory::format::nwc;
    }
  } else if (dims_size == 5) {
    if (data_format == mkldnn::memory::format::nchw) {
      return mkldnn::memory::format::ncdhw;
    } else if (data_format == mkldnn::memory::format::nhwc) {
      return mkldnn::memory::format::ndhwc;
    }
  }
  return data_format;
}

inline mkldnn::memory::format data_format_to_memory_format(
    const std::string& data_format) {
  switch (framework::StringToDataLayout(data_format)) {
    case framework::DataLayout::kNHWC:
      return mkldnn::memory::format::nhwc;
    case framework::DataLayout::kNCHW:
      return mkldnn::memory::format::nchw;
    default:
      return mkldnn::memory::format::any;
  }
}

inline mkldnn::memory::format StringToMKLDNNFormat(std::string* format) {
  std::transform(format->begin(), format->end(), format->begin(), ::tolower);

  if (!format->compare("nchw")) {
    return mkldnn::memory::format::nchw;
  } else if (!format->compare("nchw16c")) {
    return mkldnn::memory::format::nChw16c;
  } else if (!format->compare("nchw8c")) {
    return mkldnn::memory::format::nChw8c;
  } else if (!format->compare("nhwc")) {
    return mkldnn::memory::format::nhwc;
  } else {
    return mkldnn::memory::format::any;
  }
}

}  // namespace platform
}  // namespace paddle
