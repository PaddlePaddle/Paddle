/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>  // for std::move
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/jit/gen_base.h"
#include "paddle/phi/kernels/funcs/jit/kernel_base.h"
#include "paddle/phi/kernels/funcs/jit/kernel_key.h"
#include "paddle/phi/kernels/funcs/jit/kernel_pool.h"

namespace phi {
namespace jit {

class GenBase;

template <typename KernelTuple, typename PlaceType>
inline typename std::enable_if<
    std::is_same<typename KernelTuple::data_type, float>::value &&
        std::is_same<PlaceType, phi::CPUPlace>::value,
    const Kernel*>::type
GetJitCode(const typename KernelTuple::attr_type& attr) {
  using Attr = typename KernelTuple::attr_type;
  int64_t key = JitCodeKey<Attr>(attr);
  auto& codes = JitCodePool<KernelTuple::kernel_type>::Instance();
  if (codes.Has(key)) {
    return codes.AllKernels().at(key).get();
  }

  // creator is not related with attr, so can use KernelKey as key
  KernelKey kkey(KernelTuple::kernel_type, PlaceType());
  // pool: (KernelKey(type, place), vector<GenCreatorPtr>)
  auto& creator_map = JitCodeCreatorPool::Instance().AllCreators();
  auto iter = creator_map.find(kkey);
  if (iter != creator_map.end()) {
    auto& creators = iter->second;
    for (auto& cur : creators) {
      auto i = dynamic_cast<const JitCodeCreator<Attr>*>(cur.get());
      if (i && i->CanBeUsed(attr)) {
        auto p = i->CreateJitCode(attr);
        if (p) {
          auto res = p.get();
          codes.Insert(key, std::move(p));
          return res;
        }
      }
    }
  }
  return nullptr;
}

template <typename KernelTuple, typename PlaceType>
inline typename std::enable_if<
    !std::is_same<typename KernelTuple::data_type, float>::value ||
        !std::is_same<PlaceType, phi::CPUPlace>::value,
    const Kernel*>::type
GetJitCode(const typename KernelTuple::attr_type& attr UNUSED) {
  return nullptr;
}

// Refer code do not related with attr, which is just for cast
// Refer is always on CPUPlace
template <typename KernelTuple>
inline const Kernel* GetReferKernel() {
  auto& ref_pool = ReferKernelPool::Instance().AllKernels();
  KernelKey kkey(KernelTuple::kernel_type, phi::CPUPlace());
  auto ref_iter = ref_pool.find(kkey);
  PADDLE_ENFORCE_NE(
      ref_iter,
      ref_pool.end(),
      phi::errors::PreconditionNotMet(
          "Every Refer Kernel of jitcode should have reference function."));
  auto& ref_impls = ref_iter->second;
  for (auto& impl : ref_impls) {
    auto i = dynamic_cast<const ReferKernel<KernelTuple>*>(impl.get());
    if (i) {
      return i;
    }
  }
  return nullptr;
}

template <typename KernelTuple>
inline typename KernelTuple::func_type GetReferFunc() {
  auto ker = GetReferKernel<KernelTuple>();
  auto p = dynamic_cast<const ReferKernel<KernelTuple>*>(ker);
  PADDLE_ENFORCE_NOT_NULL(
      p,
      phi::errors::InvalidArgument("Get the reference code of kernel in CPU "
                                   "failed. The Refer kernel should exist."));
  return p->GetFunc();
}

// Return all Kernels that can be used
template <typename KernelTuple, typename PlaceType>
std::vector<const Kernel*> GetAllCandidateKernels(
    const typename KernelTuple::attr_type& attr) {
  // the search order should be jitcode > more > refer
  std::vector<const Kernel*> res;
  auto jitker = GetJitCode<KernelTuple, PlaceType>(attr);
  if (jitker) {
    res.emplace_back(jitker);
  }

  // more kernelpool: (KernelKey(type, place), vector<KernelPtr>)
  KernelKey kkey(KernelTuple::kernel_type, PlaceType());
  auto& pool = KernelPool::Instance().AllKernels();
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto& impls = iter->second;
    for (auto& impl : impls) {
      auto i = dynamic_cast<const KernelMore<KernelTuple>*>(impl.get());
      if (i && i->CanBeUsed(attr)) {
        res.emplace_back(i);
      }
    }
  }

  // The last implementation should be reference function on CPUPlace.
  auto ref = GetReferKernel<KernelTuple>();
  PADDLE_ENFORCE_NOT_NULL(
      ref,
      phi::errors::InvalidArgument("Get all candidate kernel in CPU failed. "
                                   "The Refer Kernel can not be empty."));
  res.emplace_back(ref);
  return res;
}

template <typename KernelTuple, typename PlaceType = phi::CPUPlace>
std::vector<std::pair<std::string, typename KernelTuple::func_type>>
GetAllCandidateFuncsWithTypes(const typename KernelTuple::attr_type& attr) {
  using Func = typename KernelTuple::func_type;
  auto kers = GetAllCandidateKernels<KernelTuple, PlaceType>(attr);
  std::vector<std::pair<std::string, Func>> res;
  for (auto k : kers) {
    std::string name = k->ImplType();
    if (name == "JitCode") {
      auto i = dynamic_cast<const GenBase*>(k);
      PADDLE_ENFORCE_NOT_NULL(i,
                              phi::errors::InvalidArgument(
                                  "Generate jitcode kernel (GenBase) failed."));
      res.emplace_back(std::make_pair(name, i->template getCode<Func>()));
    } else {
      auto i = dynamic_cast<const KernelMore<KernelTuple>*>(k);
      PADDLE_ENFORCE_NOT_NULL(
          i, phi::errors::InvalidArgument("Kernel cast (KernelMore) failed."));
      res.emplace_back(std::make_pair(name, i->GetFunc()));
    }
  }
  return res;
}

template <typename KernelTuple, typename PlaceType = phi::CPUPlace>
std::vector<typename KernelTuple::func_type> GetAllCandidateFuncs(
    const typename KernelTuple::attr_type& attr) {
  auto funcs = GetAllCandidateFuncsWithTypes<KernelTuple, PlaceType>(attr);
  std::vector<typename KernelTuple::func_type> res;
  for (auto& i : funcs) {
    res.emplace_back(i.second);
  }
  return res;
}

template <typename KernelTuple, typename PlaceType = phi::CPUPlace>
typename KernelTuple::func_type GetDefaultBestFunc(
    const typename KernelTuple::attr_type& attr) {
  auto funcs = GetAllCandidateFuncs<KernelTuple, PlaceType>(attr);
  PADDLE_ENFORCE_GE(funcs.size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The candidate jit kernel is at least one in CPU."));
  // Here could do some runtime benchmark of this attr and return the best one.
  // But yet just get the first one as the default best one,
  // which is searched in order and tuned by offline.
  return funcs[0];
}

extern std::map<size_t, std::shared_ptr<void>>& GetFuncCacheMap();

template <typename KernelTuple, typename PlaceType>
class KernelFuncs {
 public:
  KernelFuncs() = default;
  static KernelFuncs& Cache() {
    auto& func_cache_map = GetFuncCacheMap();
    auto key = typeid(KernelFuncs<KernelTuple, PlaceType>).hash_code();
    auto iter = func_cache_map.find(key);
    if (iter != func_cache_map.end()) {
      return *(KernelFuncs<KernelTuple, PlaceType>*)(iter->second.get());
    } else {
      std::shared_ptr<void> cache =
          std::make_shared<KernelFuncs<KernelTuple, PlaceType>>();
      func_cache_map.emplace(key, cache);
      return *(KernelFuncs<KernelTuple, PlaceType>*)(cache.get());
    }
  }

  // the exposed interface to use
  typename KernelTuple::func_type At(
      const typename KernelTuple::attr_type& attr) {
    // Maybe here is not good enough, not all kernels should have jitcode
    int64_t key = JitCodeKey<typename KernelTuple::attr_type>(attr);
    if (Has(key)) {
      return funcs_.at(key);
    }
    // If do not have this attr in cache then get the default best
    auto func = GetDefaultBestFunc<KernelTuple, PlaceType>(attr);
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

inline std::ostream& operator<<(std::ostream& os, const adam_attr_t& attr) {
  os << "beta1[" << attr.beta1 << "],beta2[" << attr.beta2 << "]";
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
}  // namespace phi
