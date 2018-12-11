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

#include <memory>  // for unique_ptr
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/operators/jit/gen_base.h"
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/operators/jit/kernel_key.h"
#include "paddle/fluid/operators/jit/kernel_pool.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace jit {

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

template <KernelType KT, typename T, typename Func, typename Attr,
          typename PlaceType = platform::CPUPlace>
const Func Get(Attr attr) {
  size_t key = JitCodeKey<Attr>(attr);
  auto& codes = JitCodePool<KT>().Instance();
  if (codes.Has(key)) {
    return codes.AllKernels().at(key)->template getCode<Func>();
  }

  KernelKey kkey(KT, PlaceType());
  if (std::is_same<PlaceType, platform::CPUPlace>::value) {
    // pool: (KernelKey(type, place), vector<GenCreatorPtr>)
    auto& creator_map = JitCodeCreatorPool().Instance().AllCreators();
    auto iter = creator_map.find(kkey);
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

  // pool: (KernelKey(type, place), vector<KernelPtr>)
  auto& pool = KernelPool().Instance().AllKernels();
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

}  // namespace jit
}  // namespace operators
}  // namespace paddle
