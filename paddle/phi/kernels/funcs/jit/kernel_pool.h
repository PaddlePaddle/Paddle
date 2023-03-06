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

#include <map>
#include <memory>  // for unique_ptr
#include <string>
#include <unordered_map>
#include <utility>  // for move
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/jit/gen_base.h"
#include "paddle/phi/kernels/funcs/jit/kernel_base.h"
#include "paddle/phi/kernels/funcs/jit/kernel_key.h"

namespace phi {
namespace jit {

struct KernelKey;

extern std::map<size_t, std::shared_ptr<void>>& GetJITCodesMap();

template <KernelType KT>
class JitCodePool {
  typedef std::unique_ptr<GenBase> GenBasePtr;
  typedef std::unordered_map<int64_t, GenBasePtr> JitCodeMap;

 public:
  JitCodePool() = default;
  static JitCodePool& Instance() {
    auto& jit_codes_map = GetJITCodesMap();
    auto key = typeid(JitCodePool<KT>).hash_code();
    auto iter = jit_codes_map.find(key);
    if (iter != jit_codes_map.end()) {
      return *(JitCodePool<KT>*)(iter->second.get());
    } else {
      std::shared_ptr<void> cache = std::make_shared<JitCodePool<KT>>();
      jit_codes_map.emplace(key, cache);
      return *(JitCodePool<KT>*)(cache.get());
    }
  }

  const JitCodeMap& AllKernels() { return codes_; }

  bool Has(int64_t key) const { return codes_.find(key) != codes_.end(); }

  void Insert(int64_t key, GenBasePtr value) {
    codes_.emplace(key, std::move(value));
  }

 private:
  JitCodeMap codes_;
  DISABLE_COPY_AND_ASSIGN(JitCodePool);
};

class JitCodeCreatorPool {
  typedef std::unique_ptr<const GenCreator> GenCreatorPtr;
  typedef std::
      unordered_map<KernelKey, std::vector<GenCreatorPtr>, KernelKey::Hash>
          GenCreatorPtrMap;

 public:
  JitCodeCreatorPool() = default;
  static JitCodeCreatorPool& Instance();
  GenCreatorPtrMap& AllCreators() { return creators_; }
  void Insert(const KernelKey& key, GenCreatorPtr value) {
    if (creators_.find(key) == creators_.end()) {
      creators_.emplace(key, std::vector<GenCreatorPtr>());
    }
    creators_.at(key).emplace_back(std::move(value));
  }

 private:
  GenCreatorPtrMap creators_;
  DISABLE_COPY_AND_ASSIGN(JitCodeCreatorPool);
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

}  // namespace jit
}  // namespace phi
