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

#include <functional>
#include <optional>

#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/equation_variable.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/common/enforce.h"
namespace cinn::adt {

DEFINE_ADT_TAG(tReduced);
DEFINE_ADT_TAG(tInjective);
DEFINE_ADT_UNION(ScheduleDim, tReduced<LoopSize>, tInjective<LoopSize>);

LoopSize GetLoopSize(const ScheduleDim& sched_dim);
List<int> GetReduceAxis(const List<ScheduleDim>& loop_sizes);
List<int> GetInjectiveAxis(const List<ScheduleDim>& loop_sizes);

class IGroup;

List<ScheduleDim> MakeAnchorScheduleDims(
    const IGroup& igroup,
    const std::function<Value(const Iterator&)>& Value4Iterator,
    const List<LoopSize>& loop_sizes,
    const List<Iterator>& anchor_iterators);

}  // namespace cinn::adt
