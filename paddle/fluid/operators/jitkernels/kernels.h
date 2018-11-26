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

#ifdef PADDLE_WITH_XBYAK
#include "paddle/fluid/operators/jitkernels/jitcode/jitcode.h"
#endif

namespace paddle {
namespace operators {
namespace jitkernels {

template <KernelType KT>
class JitCodePool {
 public:
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
  JitCodePool() = default;
  std::unordered_map<size_t, std::shared_ptr<const JitBase>> codes_;

  DISABLE_COPY_AND_ASSIGN(JitCodePool);
};

// std::tuple<T, Func, Attr>
template <typename T, typename Func, typename Attr>
struct KernelAttr {
  typedef T data_type;
  typedef Func return_type;
  typedef Attr attr_type;
};

class KernelPool {
 public:
  static KernelPool& Instance();

  typedef std::unique_ptr<const Kernel> KernelPtr;
  typedef std::unordered_map<KernelKey, std::vector<KernelPtr>, KernelKey::Hash>
      KernelMap;
  KernelMap& AllKernels() { return pool_; }

  void Insert(const KernelKey& key, KernelPtr value) {
    if (pool_.find(key) == pool_.end()) {
      pool_.emplace(key, std::vector<KernelPtr>());
    }
    pool_.at(key).emplace_back(std::move(value));
  }
  KernelPool() = default;

 private:
  KernelMap pool_;

  DISABLE_COPY_AND_ASSIGN(KernelPool);
};

// TODO(TJ): create_jitcode;

// TODO(TJ): make tuple? named KernelAttr
template <KernelType KT, typename T, typename Func, typename Attr,
          typename PlaceType = platform::CPUPlace>
Func Get(Attr attr) {
  size_t key = GetKey<Attr>(attr);
  auto jitcode = JitCodePool<KT>().Instance().Get(key);
  if (jitcode) {
    return jitcode->template getCode<Func>();
  }

#ifdef PADDLE_WITH_XBYAK
// // jitcode::JitCode is under protection of PADDLE_WITH_XBYAK
// if (std::is_same<PlaceType, platform::CPUPlace>::value) {
//   if (UseJitCode<KT, T, Attr>(attr)) {
//     std::shared_ptr<JitBase> p(std::make_shared<jitcode::JitCode<KT, Attr>>(
//         attr, CodeSize<KT, Attr>(attr)));
//     JitCodePool<KT>().Instance().Insert(key, p);
//     return p->getCode<Func>();
//   }
// }
#endif

  // (KernelKey(type, place), vector<Kernel>)
  auto& pool = KernelPool().Instance().AllKernels();
  KernelKey kkey(KT, PlaceType());
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto impls = iter->second;
    for (auto impl : impls) {
      auto i = std::dynamic_pointer_cast<KernelImpl<T, Func, Attr>>(impl.get());
      if (i && i->UseMe(attr)) {
        return i->GetFunc();
      }
    }
  }

  // The last implementation should be reference function on CPU
  // Every kernel should have refer code.

  //  because of test refer should have it's own pool
  // PADDLE_ENFORCE_GT(list.size(), 1) << "Should have refer implemtation";
  // const auto& refer = KernelRefer<KT, T>().AllKernels();
  // return refer.Get<Func>();

  return nullptr;
}

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
