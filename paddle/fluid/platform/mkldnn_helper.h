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

class MKLDNNHandler {
 public:
  MKLDNNHandler(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        key_(base_key),
        is_reusing_(false) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::primitive_desc mdp, void* ptr,
      const std::string& suffix) {
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((mem_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(mdp, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
      // Mark that reusing happenned. All primitives from operator instance
      // should be reused or none of them. So we check consistency
      is_reusing_ = true;
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(const mkldnn::memory::desc& md,
                                                void* ptr,
                                                const std::string& suffix) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((mem_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(
          mkldnn::memory::primitive_desc{md, engine_}, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
      // Mark that reusing happenned. All primitives from operator instance
      // should be reused or none of them. So we check consistency
      is_reusing_ = true;
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      mkldnn::memory::primitive_desc& mpd,       // NOLINT
      mkldnn::memory::primitive_desc& user_mpd,  // NOLINT
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    // create reorder primitive if the input format is not the preferred one
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto target_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((target_memory_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (target_memory_p == nullptr) {
      target_memory_p = user_memory_p;
      std::shared_ptr<mkldnn::primitive> reorder_p;
      if (mpd != user_mpd) {
        target_memory_p = std::make_shared<mkldnn::memory>(mpd);

        auto reorder_p =
            std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);
        pipeline.push_back(*reorder_p);
      }
      dev_ctx_.SetBlob(local_key, target_memory_p);
    } else {
      // Make reorder if needed
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        pipeline.push_back(*reorder_p);
      }
      is_reusing_ = true;
    }
    return target_memory_p;
  }

  static std::string GetHash(mkldnn::memory::dims& operand_dims,  // NOLINT
                             const std::string& suffix) {
    auto dims2str = [](const mkldnn::memory::dims& operand_dims) {
      std::string dstr = "";
      for (size_t i = 0; i < operand_dims.size(); ++i) {
        dstr += std::to_string(operand_dims[i]) + "-";
      }
      return dstr;
    };

    return dims2str(operand_dims) + suffix;
  }

 protected:
  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  std::string key_;
  bool is_reusing_;
};

inline mkldnn::memory::format MKLDNNFormatForSize(
    size_t dims_size, mkldnn::memory::format data_format) {
  if (dims_size == 1) {
    return mkldnn::memory::format::x;
  } else if (dims_size == 2) {
    return mkldnn::memory::format::nc;
  }
  return data_format;
}

}  // namespace platform
}  // namespace paddle
