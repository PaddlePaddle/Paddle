// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/api/op_group.h"

namespace cinn {
namespace hlir {
namespace pass {

using OpGroupPtr = api::OpGroup;

class FuseHelper {
 public:
  virtual ~FuseHelper() = default;

  virtual bool AllOutputsSameSize(const OpGroupPtr& first,
                                  const OpGroupPtr& second) const = 0;

  virtual bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src,
                                               const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseBroadcast(const OpGroupPtr& src,
                                        const OpGroupPtr& dst) const = 0;

  virtual bool HorizontalWithInjective(const OpGroupPtr& src,
                                       const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseReduce(const OpGroupPtr& src,
                                     const OpGroupPtr& dst) const = 0;

  virtual bool BroadcastFuseReduce(const OpGroupPtr& src,
                                   const OpGroupPtr& dst) const = 0;

  virtual bool InjectiveHorizontalWithReduce(const OpGroupPtr& src,
                                             const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseElementwise(const OpGroupPtr& src,
                                     const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseBroadcast(const OpGroupPtr& src,
                                   const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseReduce(const OpGroupPtr& src,
                                const OpGroupPtr& dst) const = 0;

  virtual bool IsReachable(const OpGroupPtr& lhs,
                           const OpGroupPtr& rhs) const = 0;

  virtual bool DetectCycleIfFuse(const OpGroupPtr& src,
                                 const OpGroupPtr& dst) const = 0;

  virtual bool IsConsumerSetsReachable(
      const OpGroupPtr& group,
      const std::unordered_set<OpGroupPtr>& consumers) const = 0;

 protected:
  FuseHelper() = default;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
