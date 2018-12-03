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

#include <memory>  // for shared_ptr
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/operators/jitkernels/jitcode_base.h"
#include "paddle/fluid/operators/jitkernels/kernel_base.h"
#include "paddle/fluid/operators/jitkernels/kernel_key.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace jitkernels {

// TODO(TJ): rename file to kernel_pool

template <KernelType KT>
class JitCodePool {
 public:
  JitCodePool() = default;
  static JitCodePool& Instance() {
    static thread_local JitCodePool<KT> g_jit_codes;
    return g_jit_codes;
  }

  std::shared_ptr<const JitBase> Get(size_t key) const {
    if (codes_.find(key) == codes_.end()) {
      return nullptr;
    }
    return codes_.at(key);
  }

  void Insert(size_t key, const std::shared_ptr<const JitBase>& value) {
    codes_.insert({key, value});
  }

 private:
  std::unordered_map<size_t, std::shared_ptr<const JitBase>> codes_;
  DISABLE_COPY_AND_ASSIGN(JitCodePool);
};

// TODO(TJ): std::tuple<T, Func, Attr>
template <typename T, typename Func, typename Attr>
struct KernelAttr {
  typedef T data_type;
  typedef Func return_type;
  typedef Attr attr_type;
};

typedef std::unique_ptr<const Kernel> KernelPtr;
typedef std::unordered_map<KernelKey, std::vector<KernelPtr>, KernelKey::Hash>
    KernelMap;

class KernelPool {
 public:
  static KernelPool& Instance();
  KernelPool() = default;
  KernelMap& AllKernels() { return pool_; }
  void Insert(const KernelKey& key, KernelPtr value) {
    if (pool_.find(key) == pool_.end()) {
      pool_.emplace(key, std::vector<KernelPtr>());
    }
    pool_.at(key).emplace_back(std::move(value));
  }

 private:
  KernelMap pool_;
  DISABLE_COPY_AND_ASSIGN(KernelPool);
};

// Every kernel should have refer code and it should be used in unit tests,
// so refer kernels should have it's independent kernel pool
class ReferKernelPool {
 public:
  static ReferKernelPool& Instance();
  ReferKernelPool() = default;
  KernelMap& AllKernels() { return pool_; }
  void Insert(const KernelKey& key, KernelPtr value) {
    if (pool_.find(key) == pool_.end()) {
      pool_.emplace(key, std::vector<KernelPtr>());
    }
    pool_.at(key).emplace_back(std::move(value));
  }

 private:
  KernelMap pool_;
  DISABLE_COPY_AND_ASSIGN(ReferKernelPool);
};

// Refer code do not related with attr, and always on CPUPlace
template <KernelType KT, typename T, typename Func, typename Attr>
inline Func GetRefer() {
  auto& ref_pool = ReferKernelPool().Instance().AllKernels();
  KernelKey kkey(KT, platform::CPUPlace());
  auto ref_iter = ref_pool.find(kkey);
  PADDLE_ENFORCE(ref_iter != ref_pool.end(),
                 "Every Kernel should have reference function.");
  auto& ref_impls = ref_iter->second;
  for (auto& impl : ref_impls) {
    auto i = dynamic_cast<const ReferKernel<T, Func, Attr>*>(impl.get());
    if (i) {
      return i->GetFunc();
    }
  }
  return nullptr;
}

// TODO(TJ): make tuple? named KernelAttr
template <KernelType KT, typename T, typename Func, typename Attr,
          typename PlaceType = platform::CPUPlace>
Func Get(Attr attr) {
  // size_t key = GetKey<Attr>(attr);
  // auto jitcode = JitCodePool<KT>().Instance().Get(key);
  // if (jitcode) {
  //   return jitcode->template getCode<Func>();
  // }

  if (std::is_same<PlaceType, platform::CPUPlace>::value &&
      std::is_same<T, float>::value) {  // TODO(TJ): float move to create
    // auto p = CreateJitCode<KT, Attr>(attr);
    // if (p) {
    //   JitCodePool<KT>().Instance().Insert(key, p);
    //   return p->template getCode<Func>();
    // }
  }

  // pool: (KernelKey(type, place), vector<Kernel>)
  auto& pool = KernelPool().Instance().AllKernels();
  KernelKey kkey(KT, PlaceType());
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto& impls = iter->second;
    for (auto& impl : impls) {
      auto i = dynamic_cast<const KernelImpl<T, Func, Attr>*>(impl.get());
      if (i && i->UseMe(attr)) {
        return i->GetFunc();
      }
    }
  }

  // The last implementation should be reference function on CPUPlace.
  return GetRefer<KT, T, Func, Attr>();
}

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
