// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/types.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {

using phi::distributed::AllreduceOptions;
using phi::distributed::BarrierOptions;
using phi::distributed::BroadcastOptions;
using phi::distributed::CommType;
using phi::distributed::GatherOptions;
using phi::distributed::GetPartialTensor;
using phi::distributed::ReduceOp;
using phi::distributed::ReduceOptions;
using phi::distributed::ReduceScatterOptions;
using phi::distributed::ScatterOptions;
constexpr int kIgnoreId = -1;

using phi::distributed::ProcessGroup;
using phi::distributed::ProcessGroupIdMap;
using phi::distributed::ProcessGroupMapFromGid;

static void CheckTensorContiguous(const phi::DenseTensor& tensor) {
  if (!tensor.meta().is_contiguous()) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The tensor must be contiguous"));
  }
}

static void CheckTensorContiguous(const std::vector<phi::DenseTensor>& inputs) {
  for (const auto& tensor : inputs) {
    if (!tensor.meta().is_contiguous()) {
      PADDLE_THROW(
          common::errors::InvalidArgument("The tensor must be contiguous"));
    }
  }
}

}  //  namespace distributed
}  //  namespace paddle
