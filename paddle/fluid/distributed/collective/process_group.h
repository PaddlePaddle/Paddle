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

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
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

static void CheckTensorSamePlace(const std::vector<phi::DenseTensor>& tensors) {
  for (const auto& tensor : tensors) {
    if (tensor.place() != tensors[0].place()) {
      PADDLE_THROW(
          common::errors::InvalidArgument("The tensors must be in the same "
                                          "place"));
    }
  }
}

static std::vector<int64_t> GetAllToAllSplitSizes(
    const std::vector<phi::DenseTensor>& tensors) {
  std::vector<int64_t> split_sizes(tensors.size());
  std::transform(tensors.begin(),
                 tensors.end(),
                 split_sizes.begin(),
                 [](const phi::DenseTensor& tensor) { return tensor.numel(); });
  return split_sizes;
}

static std::vector<const void*> GetTensorPtrs(
    const std::vector<phi::DenseTensor>& tensors) {
  std::vector<const void*> tensor_ptrs(tensors.size());
  std::transform(tensors.begin(),
                 tensors.end(),
                 tensor_ptrs.begin(),
                 [](const phi::DenseTensor& tensor) { return tensor.data(); });
  return tensor_ptrs;
}

static int64_t GetTensorNumel(const std::vector<phi::DenseTensor>& tensors) {
  return std::accumulate(tensors.begin(),
                         tensors.end(),
                         int64_t(0),
                         [](int64_t sum, const phi::DenseTensor& tensor) {
                           return sum + tensor.numel();
                         });
}

}  //  namespace distributed
}  //  namespace paddle
