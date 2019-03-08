/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

extern "C" {
#include <xxhash.h>
}
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>  // for std::move
#include <vector>
#include "paddle/fluid/operators/jit/gen_base.h"
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/operators/jit/kernel_key.h"
#include "paddle/fluid/operators/jit/kernel_pool.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace jit {

template <typename KernelTuple, typename PlaceType>
inline typename std::enable_if<
    std::is_same<typename KernelTuple::data_type, float>::value &&
        std::is_same<PlaceType, platform::CPUPlace>::value,
    typename KernelTuple::func_type>::type
GetJitCode(const typename KernelTuple::attr_type& attr) {
  using Func = typename KernelTuple::func_type;
  using Attr = typename KernelTuple::attr_type;
  size_t key = JitCodeKey<Attr>(attr);
  auto& codes = JitCodePool<KernelTuple::kernel_type>().Instance();
  if (codes.Has(key)) {
    return codes.AllKernels().at(key)->template getCode<Func>();
  }

  // creator is not related with attr, so can use KernelKey as key
  KernelKey kkey(KernelTuple::kernel_type, PlaceType());
  // pool: (KernelKey(type, place), vector<GenCreatorPtr>)
  auto& creator_map = JitCodeCreatorPool().Instance().AllCreators();
  auto iter = creator_map.find(kkey);
  if (iter != creator_map.end()) {
    auto& creators = iter->second;
    for (auto& cur : creators) {
      auto i = dynamic_cast<const JitCodeCreator<Attr>*>(cur.get());
      if (i && i->UseMe(attr)) {
        auto p = i->CreateJitCode(attr);
        if (p) {
          auto f = p->template getCode<Func>();
          codes.Insert(key, std::move(p));
          return f;
        }
      }
    }
  }
  return nullptr;
}

template <typename KernelTuple, typename PlaceType>
inline typename std::enable_if<
    !std::is_same<typename KernelTuple::data_type, float>::value ||
        !std::is_same<PlaceType, platform::CPUPlace>::value,
    typename KernelTuple::func_type>::type
GetJitCode(const typename KernelTuple::attr_type& attr) {
  return nullptr;
}

// Refer code do not related with attr, which is just for cast
// Refer is always on CPUPlace
template <typename KernelTuple>
inline typename KernelTuple::func_type GetRefer() {
  auto& ref_pool = ReferKernelPool().Instance().AllKernels();
  KernelKey kkey(KernelTuple::kernel_type, platform::CPUPlace());
  auto ref_iter = ref_pool.find(kkey);
  PADDLE_ENFORCE(ref_iter != ref_pool.end(),
                 "Every Kernel should have reference function.");
  auto& ref_impls = ref_iter->second;
  for (auto& impl : ref_impls) {
    auto i = dynamic_cast<const ReferKernel<KernelTuple>*>(impl.get());
    if (i) {
      return i->GetFunc();
    }
  }
  return nullptr;
}

template <typename KernelTuple, typename PlaceType = platform::CPUPlace>
typename KernelTuple::func_type Get(
    const typename KernelTuple::attr_type& attr) {
  auto jitfunc = GetJitCode<KernelTuple, PlaceType>(attr);
  if (jitfunc) {
    return jitfunc;
  }

  // pool: (KernelKey(type, place), vector<KernelPtr>)
  KernelKey kkey(KernelTuple::kernel_type, PlaceType());
  auto& pool = KernelPool().Instance().AllKernels();
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto& impls = iter->second;
    for (auto& impl : impls) {
      auto i = dynamic_cast<const KernelMore<KernelTuple>*>(impl.get());
      if (i && i->UseMe(attr)) {
        return i->GetFunc();
      }
    }
  }

  // The last implementation should be reference function on CPUPlace.
  return GetRefer<KernelTuple>();
}

template <typename KernelTuple, typename PlaceType>
class KernelFuncs {
 public:
  KernelFuncs() = default;
  static KernelFuncs& Cache() {
    static thread_local KernelFuncs<KernelTuple, PlaceType> g_func_cache;
    return g_func_cache;
  }

  // the exposed interface to use
  typename KernelTuple::func_type At(
      const typename KernelTuple::attr_type& attr) {
    // XXH64: 13.8 GB/s
    // TODO(TJ): change me, maybe not all attr change need one key, should be
    // attrkey
    int64_t key = XXH64(&attr, sizeof(typename KernelTuple::attr_type), 0);
    if (Has(key)) {
      return funcs_.at(key);
    }
    // If do not have this attr in cache,
    // then could run some runtime benchmark of this attr and save the best one.
    // Here just get the offline benchmarked best one.
    auto func = Get<KernelTuple, PlaceType>(attr);
    Insert(key, func);
    return func;
  }

  typename KernelTuple::func_type operator[](
      const typename KernelTuple::attr_type& attr) {
    return At(attr);
  }

 protected:
  bool Has(int64_t key) const { return funcs_.find(key) != funcs_.end(); }

  void Insert(int64_t key, typename KernelTuple::func_type func) {
    funcs_.emplace(key, func);
  }

 private:
  std::unordered_map<int64_t, typename KernelTuple::func_type> funcs_;
  DISABLE_COPY_AND_ASSIGN(KernelFuncs);
};

const char* to_string(KernelType kt);
const char* to_string(SeqPoolType kt);

KernelType to_kerneltype(const std::string& act);

inline std::ostream& operator<<(std::ostream& os, const lstm_attr_t& attr) {
  os << "dim_size[" << attr.d << "],act_gate[" << to_string(attr.act_gate)
     << "],act_cand[" << to_string(attr.act_cand) << "],act_cell["
     << to_string(attr.act_cell) << "],use_peephole["
     << (attr.use_peephole ? "True" : "False") << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const gru_attr_t& attr) {
  os << "dim_size[" << attr.d << "],act_gate[" << to_string(attr.act_gate)
     << "],act_cand[" << to_string(attr.act_cand) << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const seq_pool_attr_t& attr) {
  os << "height_size[" << attr.h << "],width_size[" << attr.w << "],pool_type["
     << to_string(attr.type) << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const emb_seq_pool_attr_t& attr) {
  os << "table_height[" << attr.table_height << "],table_width["
     << attr.table_width << "],index_height[" << attr.index_height
     << "],index_width[" << attr.index_width << "],output_width["
     << attr.out_width << "],pool_type[" << to_string(attr.pool_type) << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const sgd_attr_t& attr) {
  os << "param_height[" << attr.param_height << "],param_width["
     << attr.param_width << "],grad_height[" << attr.grad_height
     << "],grad_width[" << attr.grad_width << "],selected_rows_size["
     << attr.selected_rows_size << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const matmul_attr_t& attr) {
  os << "M[" << attr.m << "],N[" << attr.n << "],K[" << attr.k << "]";
  return os;
}

// expose the method to pack matmul weight
template <typename T>
void pack_weights(const T* src, T* dst, int n, int k);

}  // namespace jit
}  // namespace operators
}  // namespace paddle
