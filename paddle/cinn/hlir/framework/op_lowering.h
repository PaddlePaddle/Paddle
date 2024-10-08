// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/hlir/framework/op_lowering_impl_base.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/lang/packed_func.h"
#ifndef CINN_WITH_ONLY
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"
#endif

namespace cinn {
namespace hlir {
namespace framework {

using cinn::common::Target;

template <typename T>
class OpLowerer {
 public:
  explicit OpLowerer(OpLowererImplBase<T>* impl) { impl_.reset(impl); }
  ~OpLowerer() {}

  std::vector<ir::LoweredFunc> Lower(const T& group,
                                     bool apply_op_schedule = true,
                                     bool apply_group_schedule = true,
                                     bool apply_pass = true) {
    return impl_->Lower(
        group, apply_op_schedule, apply_group_schedule, apply_pass);
  }

  BucketLoweredFuncsWrapper BucketLower(const T& group) {
    return impl_->BucketLower(group);
  }

 private:
  std::shared_ptr<OpLowererImplBase<T>> impl_;
};

#ifndef CINN_WITH_ONLY
template <typename T = pir::OpLoweringGroupPtr>
OpLowerer<T> CreateOpLowerer(const Target&);

template <>
inline OpLowerer<pir::OpLoweringGroupPtr> CreateOpLowerer(
    const Target& target) {
  auto* impl_base = new pir::OpLowererImpl(target);
  return OpLowerer<pir::OpLoweringGroupPtr>(impl_base);
}
#endif

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
