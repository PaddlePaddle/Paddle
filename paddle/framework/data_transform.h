/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <functional>
#include <utility>
#include <vector>

#include "paddle/framework/op_kernel_type.h"
#include "paddle/framework/tensor.h"
#include "paddle/framework/variable.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

using DataTransformFN =
    std::function<void(const std::vector<platform::DeviceContext*> ctx,
                       const Variable& in, Variable* out)>;
using KernelTypePair = std::pair<OpKernelType, OpKernelType>;

static void hash_combine(std::size_t& seed, const OpKernelType& t) {
  OpKernelType::Hash kernel_type_hasher;
  seed ^= kernel_type_hasher(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct KernelTypePairHash {
  size_t operator()(const KernelTypePair& kernel_pair) const {
    std::size_t seed = 0;
    hash_combine(seed, kernel_pair.first);
    hash_combine(seed, kernel_pair.second);

    return seed;
  }
};

using DataTransformMap =
    std::unordered_map<KernelTypePair, DataTransformFN, KernelTypePairHash>;

class DataTransformFnMap {
 public:
  static DataTransformFnMap& Instance();

  bool Has(const KernelTypePair& key_pair) const {
    return map_.find(key_pair) != map_.end();
  }

  void Insert(const OpKernelType& left, const OpKernelType& right,
              const DataTransformFN& data_tranform_fn) {
    Insert(std::make_pair(left, right), data_tranform_fn);
  }

  void Insert(const KernelTypePair& kernel_type_pair,
              const DataTransformFN& data_tranform_fn) {
    PADDLE_ENFORCE(!Has(kernel_type_pair),
                   "KernelTypePair %s has been registered", "");
    map_.insert({kernel_type_pair, data_tranform_fn});
  }

  const DataTransformFN& Get(const KernelTypePair& key_pair) const {
    auto data_transformer = GetNullable(key_pair);
    PADDLE_ENFORCE_NOT_NULL(data_transformer,
                            "DataTransformFN should not be NULL");
    return *data_transformer;
  }

  const DataTransformFN* GetNullable(const KernelTypePair& key_pair) const {
    auto it = map_.find(key_pair);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &(it->second);
    }
  }

  const DataTransformMap& Map() const { return map_; }

 private:
  DataTransformFnMap() = default;
  DataTransformMap map_;
  DISABLE_COPY_AND_ASSIGN(DataTransformFnMap);
};

// generate unique name with __LINE__
// refs https://stackoverflow.com/questions/1597007
#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)
#define REGISTER_DATA_TRANSFORM_FN(from, to, fn)                              \
  static int TOKENPASTE2(fn_, __LINE__)() {                                   \
    ::paddle::framework::DataTransformFnMap::Instance().Insert(from, to, fn); \
    return 0;                                                                 \
  }                                                                           \
  static int TOKENPASTE2(var_, __LINE__) __attribute__((unused)) =            \
      TOKENPASTE2(fn_, __LINE__)()

}  // namespace framework
}  // namespace paddle
