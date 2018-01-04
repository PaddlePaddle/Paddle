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
#include "paddle/operators/math/math_function.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/macros.h"
#include "paddle/platform/transform.h"

namespace paddle {
namespace framework {

using KernelTypePair = std::pair<OpKernelType, OpKernelType>;

using DataTransformFn =
    std::function<void(const platform::DeviceContext*, const KernelTypePair&,
                       const Variable&, Variable*)>;

struct KernelTypePairHash {
  static void HashCombine(const OpKernelType& t, std::size_t* seed) {
    OpKernelType::Hash kernel_type_hasher;
    (*seed) ^= kernel_type_hasher(t) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }

  size_t operator()(const KernelTypePair& kernel_pair) const {
    std::size_t seed = 0;
    HashCombine(kernel_pair.first, &seed);
    HashCombine(kernel_pair.second, &seed);
    return seed;
  }
};

template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

template <typename InType>
struct CastDataType {
  CastDataType(const framework::Tensor& in, framework::Tensor* out,
               const platform::DeviceContext* ctx)
      : in_(in), out_(out), ctx_(ctx) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;

  template <typename OutType>
  void operator()() {
    auto place = ctx_->GetPlace();

    auto* in_begin = in_.data<InType>();
    auto numel = in_.numel();
    auto* in_end = in_begin + numel;
    auto* out_begin = out_->mutable_data<OutType>(place);

    if (platform::is_cpu_place(place)) {
      platform::Transform<platform::CPUDeviceContext> trans;
      auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
      trans(*context, in_begin, in_end, out_begin,
            CastDataTypeFunctor<InType, OutType>());
    } else {
      // TODO(dzhwinter): enhance CopyFrom CPU<->GPU with different data type?
      PADDLE_THROW("Unsupport CPU <-> GPU!");
    }
  }
};

struct CastDataLayout {
  CastDataLayout(const platform::DeviceContext* ctx,
                 const std::vector<int>& axis, const framework::Tensor& in,
                 framework::Tensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void operator()() {
    auto place = ctx_->GetPlace();

    if (platform::is_cpu_place(place)) {
      operators::math::Transpose<platform::CPUDeviceContext, T, 4> trans4;
      auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
      trans4(*context, in_, out_, axis_);
    } else {
      PADDLE_THROW("Unsupport CPU <-> GPU!");
    }
  }
};

using DataTransformMap =
    std::unordered_map<KernelTypePair, DataTransformFn, KernelTypePairHash>;

class DataTransformFnMap {
 public:
  static DataTransformFnMap& Instance();

  bool Has(const KernelTypePair& key_pair) const {
    return map_.find(key_pair) != map_.end();
  }

  void Insert(const OpKernelType& left, const OpKernelType& right,
              const DataTransformFn& data_tranform_fn) {
    Insert(std::make_pair(left, right), data_tranform_fn);
  }

  void Insert(const KernelTypePair& kernel_type_pair,
              const DataTransformFn& data_tranform_fn) {
    PADDLE_ENFORCE(!Has(kernel_type_pair),
                   "KernelTypePair %s has been registered", "");
    map_.insert({kernel_type_pair, data_tranform_fn});
  }

  const DataTransformFn& Get(const KernelTypePair& key_pair) const {
    auto data_transformer = GetNullable(key_pair);
    PADDLE_ENFORCE_NOT_NULL(data_transformer,
                            "DataTransformFn should not be NULL");
    return *data_transformer;
  }

  const DataTransformFn* GetNullable(const KernelTypePair& key_pair) const {
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
