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
#include "paddle/platform/device_context.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

using DataTransformFN = std::function<void(
    std::vector<platform::DeviceContext*> ctx, const Tensor& in, Tensor* out)>;
using KernelTypePair = std::pair<OpKernelType, OpKernelType>;

struct KernelTypePairHash {
  size_t operator()(const KernelTypePair& kernel_pair) const {
    OpKernelType::Hash kernel_type_hasher;
    size_t left_hasher = kernel_type_hasher(kernel_pair.first) << 1;
    size_t right_hasher = kernel_type_hasher(kernel_pair.second);
    std::hash<size_t> hasher;
    return hasher(left_hasher + right_hasher);
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

#define REGISTER_DATA_TRANSFORM_FN(uniq_name, from, to, fn)                   \
  int uniq_name##_fn() {                                                      \
    ::paddle::framework::DataTransformFnMap::Instance().Insert(from, to, fn); \
    return 0;                                                                 \
  }                                                                           \
  static int uniq_name##_var __attribute__((unused)) = uniq_name##_fn()

}  // namespace framework
}  // namespace paddle
