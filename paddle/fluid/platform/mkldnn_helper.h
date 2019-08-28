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

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int64_t>& dims,
                                          mkldnn::memory::data_type data_type,
                                          mkldnn::memory::format_tag format) {
  mkldnn::memory::dims tz = dims;
  return mkldnn::memory::desc({tz}, data_type, format);
}

inline bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_mkldnn = ctx.Attr<bool>("use_mkldnn");
  return use_mkldnn && platform::is_cpu_place(ctx.GetPlace());
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

inline void Reorder(mkldnn::memory& src, mkldnn::memory& dst,
                    const mkldnn::engine& engine) {
  auto reorder_prim = mkldnn::reorder(src, dst);
  mkldnn::stream astream(engine);
  reorder_prim.execute(astream, src, dst);
  astream.wait();
}

// TODO(grygielski)
inline mkldnn::memory::format_tag GetMKLDNNFormat(
    mkldnn::memory::desc mem_desc) {
  // TODO(grygielski) clean this mess
  // mkldnn::memory::desc mem_desc({1, 64, 128, 128},
  // mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nChw16c);
  // auto mem_desc = memory.get_desc();
  // std::cout<<"nDIMS:"<<mem_desc.data.ndims<<std::endl;
  // std::cout<<"DIMS:"<<mem_desc.data.dims[0]<<" "<<mem_desc.data.dims[1]<<"
  // "<<mem_desc.data.dims[2]<<" "<<mem_desc.data.dims[3]<<std::endl;
  // std::cout<<"STRIDES:"<<mem_desc.data.format_desc.blocking.strides[0]<<"
  // "<<mem_desc.data.format_desc.blocking.strides[1]<<"
  // "<<mem_desc.data.format_desc.blocking.strides[2]<<"
  // "<<mem_desc.data.format_desc.blocking.strides[3]<<std::endl;
  // std::cout<<"INNER
  // BLOCKS:"<<mem_desc.data.format_desc.blocking.inner_blks[0]<<std::endl;
  // std::cout<<"INNER BLOCK
  // IDX:"<<mem_desc.data.format_desc.blocking.inner_idxs[0]<<std::endl;
  // if(mem_desc.data.ndims < 2) {
  //   return mkldnn::memory::format_tag::x;
  // }
  // std::vector<int64_t> mem_dims(std::begin(mem_desc.data.dims),
  // std::end(mem_desc.data.dims));
  // mkldnn::memory::desc temp(mem_dims, mkldnn::memory::data_type::f32,
  // mkldnn::memory::format_tag::nchw);
  // if(temp == mem_desc) {
  //   return mkldnn::memory::format_tag::nchw;
  // }
  // else {
  //   return mkldnn::memory::format_tag::nhwc;
  // }

  auto ndims = mem_desc.data.ndims;
  auto strides = mem_desc.data.format_desc.blocking.strides;
  auto inner_nblks = mem_desc.data.format_desc.blocking.inner_nblks;
  auto inner_blks = mem_desc.data.format_desc.blocking.inner_blks;
  auto inner_idxs = mem_desc.data.format_desc.blocking.inner_idxs;

  // TODO(grygielski) incomplete
  if (ndims == 1) {
    return mkldnn::memory::format_tag::x;
  } else if (ndims == 2) {
    if (inner_nblks == 0) {
      if (strides[0] > strides[1]) {
        return mkldnn::memory::format_tag::nc;
      } else {
        return mkldnn::memory::format_tag::cn;
      }
    }
  } else if (ndims == 3) {
    if (inner_nblks == 0) {
      if (strides[0] > strides[1] && strides[1] > strides[2]) {
        return mkldnn::memory::format_tag::ncw;
      } else {
        return mkldnn::memory::format_tag::nwc;
      }
    }
  } else if (ndims == 4) {
    if (inner_nblks == 0) {
      if (strides[0] > strides[1] && strides[1] > strides[2] &&
          strides[2] > strides[3]) {
        return mkldnn::memory::format_tag::nchw;
      } else {
        return mkldnn::memory::format_tag::nhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw16c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw8c;
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
        if(inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return mkldnn::memory::format_tag::OIhw16i16o;
        }
      }
    }
  } else if (ndims == 5) {
    if (inner_nblks == 0) {
      if (strides[0] > strides[1] && strides[1] > strides[2] &&
          strides[2] > strides[3] && strides[3] > strides[4]) {
        return mkldnn::memory::format_tag::ncdhw;
      } else {
        return mkldnn::memory::format_tag::ndhwc;
      }
    }
  }
  std::cout<<"@@@@@@@@@@ UNDEFINED FORMAT @@@@@@@@@@@@@@@@@@@"<<std::endl;
  std::cout<<"NDIMS: "<<ndims<<std::endl;
  std::cout<<"INNER_NBLKS: "<<inner_nblks<<std::endl;
  for (int i=0;i<ndims;++i) {
    std::cout<<"STRIDE["<<i<<"]: "<<strides[i]<<std::endl;
  }
  for (int i=0;i<inner_nblks;++i) {
    std::cout<<"INNER_BLKS["<<i<<"]: "<<inner_blks[i]<<std::endl;
  }
  for (int i=0;i<inner_nblks;++i) {
    std::cout<<"INNER_IDXS["<<i<<"]: "<<inner_idxs[i]<<std::endl;
  }
  return mkldnn::memory::format_tag::undef;
  // return mkldnn::memory::format_tag::nChw16c;

  // return static_cast<mkldnn::memory::format_tag>(
  //     memory.get_primitive_desc().desc().data.format_tag);
}

inline mkldnn::memory::format_tag GetMKLDNNFormat(const mkldnn::memory memory) {
  auto mem_desc = memory.get_desc();
  return GetMKLDNNFormat(mem_desc);
}

// TODO(grygielski) innecessary
// inline mkldnn::memory::format_tag GetMKLDNNFormat(
//     const mkldnn::sum::primitive_desc& memory) {
//      return mkldnn::memory::format_tag::nchw;
// return static_cast<mkldnn::memory::format_tag>(
//     memory.dst_primitive_desc().desc().data.format_tag);
//}

inline mkldnn::memory::format_tag MKLDNNFormatForSize(
    size_t dims_size, mkldnn::memory::format_tag data_format) {
  if (dims_size == 1) {
    return mkldnn::memory::format_tag::x;
  } else if (dims_size == 2) {
    return mkldnn::memory::format_tag::nc;
  } else if (dims_size == 3) {
    if (data_format == mkldnn::memory::format_tag::nchw) {
      return mkldnn::memory::format_tag::ncw;
    } else if (data_format == mkldnn::memory::format_tag::nhwc) {
      return mkldnn::memory::format_tag::nwc;
    }
  } else if (dims_size == 5) {
    if (data_format == mkldnn::memory::format_tag::nchw) {
      return mkldnn::memory::format_tag::ncdhw;
    } else if (data_format == mkldnn::memory::format_tag::nhwc) {
      return mkldnn::memory::format_tag::ndhwc;
    }
  }
  return data_format;
}

inline mkldnn::memory::format_tag data_format_to_memory_format(
    const std::string& data_format) {
  switch (framework::StringToDataLayout(data_format)) {
    case framework::DataLayout::kNHWC:
      return mkldnn::memory::format_tag::nhwc;
    case framework::DataLayout::kNCHW:
      return mkldnn::memory::format_tag::nchw;
    default:
      return mkldnn::memory::format_tag::any;
  }
}

inline mkldnn::memory::format_tag StringToMKLDNNFormat(std::string* format) {
  std::transform(format->begin(), format->end(), format->begin(), ::tolower);

  if (!format->compare("nchw")) {
    return mkldnn::memory::format_tag::nchw;
  } else if (!format->compare("nchw16c")) {
    return mkldnn::memory::format_tag::nChw16c;
  } else if (!format->compare("nchw8c")) {
    return mkldnn::memory::format_tag::nChw8c;
  } else if (!format->compare("nhwc")) {
    return mkldnn::memory::format_tag::nhwc;
  } else {
    return mkldnn::memory::format_tag::any;
  }
}

}  // namespace platform
}  // namespace paddle
