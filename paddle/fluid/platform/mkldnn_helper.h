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

#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
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
tf_pd<Type> MKLDNNBwdPrimitiveDesc(const Engine& e,
                                   const Primitive& p,
                                   Args&&... args) {
  auto desc = tf_desc<Type>(args...);
  return tf_pd<Type>(desc, e, p);
}

inline void MatchShapeToLayout(phi::DenseTensor* tensor_in,
                               phi::DataLayout from,
                               phi::DataLayout to) {
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
    VLOG(3) << "Keeping kMKLDNN/kNHWC/kNDHWC output_shape"
            << print_dims(phi::vectorize<int>(tensor_in->dims()));
    return;
  }

  switch (from) {
    case phi::DataLayout::kMKLDNN:
      if ((to == phi::DataLayout::kNHWC) || (to == phi::DataLayout::kNDHWC)) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kMKLDNN to: kNHWC/kNDHWC output_shape"
                << print_dims(dims);
      }
      break;
    case phi::DataLayout::kNHWC:
    case phi::DataLayout::kNDHWC:
      if (to == phi::DataLayout::kMKLDNN) {
        auto dims = phi::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
        tensor_in->Resize(phi::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kNHWC/kNDHWC to: kMKLDNN output_shape"
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

inline void Reorder(dnnl::memory src,
                    dnnl::memory dst,
                    const dnnl::engine& engine) {
  auto reorder_prim = dnnl::reorder(src, dst);
  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  platform::RecordEvent record_reorder("int_reorder",
                                       platform::TracerEventType::UserDefined,
                                       2,
                                       platform::EventRole::kUniqueOp);
  reorder_prim.execute(astream, src, dst);
  astream.wait();
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

inline void RegisterModelLayout(
    std::vector<std::unique_ptr<framework::OperatorBase>>& ops,  // NOLINT
    const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    // If there is already registered NHWC then quit this call
    // not to overwrite setting with analysis of internal "while" op block
    if (platform::MKLDNNDeviceContext::tls().get_cur_paddle_data_layout() ==
        phi::DataLayout::kNHWC)
      return;

    VLOG(4) << "RegisterModelLayout for mkldnn";
    auto check_attrib = [](std::unique_ptr<framework::OperatorBase>& op,
                           const std::string& attrib_name) -> bool {
      if (op->HasAttr(attrib_name)) {
        auto data_format = op->Attr<std::string>(attrib_name);
        platform::MKLDNNDeviceContext::tls().set_cur_paddle_data_layout(
            data_format.compare("NHWC") == 0 ? phi::DataLayout::kNHWC
                                             : phi::DataLayout::kNCHW);
        return true;
      } else {
        return false;
      }
    };

    for (auto& op : ops) {
      if (check_attrib(op, std::string("data_format"))) {
        return;
      }
      if (check_attrib(op, std::string("data_layout"))) {
        return;
      }
    }
  }
}

inline bool HasOpINT8DataType(const paddle::framework::OpDesc* op) {
  return (op->GetAttrIfExists<std::string>("mkldnn_data_type") == "int8" ||
          op->GetAttrIfExists<bool>("use_quantizer"));
}

inline bool HasOpBFLOAT16DataType(const paddle::framework::OpDesc* op) {
  return op->GetAttrIfExists<std::string>("mkldnn_data_type") == "bfloat16";
}

enum class RNNReorderType { PP_NTC, PP_TNC, NTC_PP, TNC_PP };

}  // namespace platform

inline std::string FindInputNameByVarName(framework::OpDesc* op,
                                          const std::string& searched_name) {
  std::string ret;
  for (const auto& name : op->InputNames())
    for (const auto& input_name : op->Input(name))
      if (input_name == searched_name) ret = name;
  return ret;
}

inline std::string FindOutputNameByVarName(framework::OpDesc* op,
                                           const std::string& searched_name) {
  std::string ret;
  for (const auto& name : op->OutputNames())
    for (const auto& output_name : op->Output(name))
      if (output_name == searched_name) ret = name;
  return ret;
}
}  // namespace paddle
