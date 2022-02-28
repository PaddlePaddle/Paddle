// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/reducer.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace distributed {

std::vector<std::vector<size_t>> Eager_AssignGroupBySize(
    const std::vector<Tensor> tensors,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices) {
  PADDLE_ENFORCE_EQ(
      tensors.size(), is_sparse_gradient.size(),
      platform::errors::PreconditionNotMet(
          "tensors len must be equal to is_sparse_gradient len, but "
          "[%lu] != [%lu]",
          tensors.size(), is_sparse_gradient.size()));
  auto check_perm = [](const std::vector<int64_t> &x) -> bool {
    size_t len = x.size();
    std::vector<size_t> cnt(len, 0);
    for (size_t i = 0; i < len; ++i) {
      if (x[i] >= static_cast<int64_t>(len) || x[i] < 0 || cnt[x[i]]) {
        return false;
      }
      cnt[x[i]]++;
    }
    return true;
  };

  PADDLE_ENFORCE_EQ(true, check_perm(tensor_indices),
                    platform::errors::PreconditionNotMet(
                        "tensor_indices must be a permutation from 0 to %lu",
                        tensor_indices.size()));
  // the return vector
  std::vector<std::vector<size_t>> res;

  // Key: the var type
  // Value: should use which index in group_size_limits for group size limit
  std::map<experimental::DataType, size_t> group_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this group>
  std::map<experimental::DataType, std::pair<std::vector<size_t>, size_t>>
      next_group;

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &var = tensors[i];

    size_t tensor_real_index = i;
    if (!tensor_indices.empty()) {
      tensor_real_index = tensor_indices[i];
    }

    if (is_sparse_gradient[tensor_real_index]) {
      // we keep sparse var a single group
      res.push_back({tensor_real_index});
      continue;
    }

    const auto &var_dtype = var.dtype();
    VLOG(3) << "var[" << var.name() << "] 's type is " << var_dtype;
    auto &group_info = next_group[var_dtype];

    int64_t var_size = -1;

    if (var.is_dense_tensor()) {
      var_size =
          std::dynamic_pointer_cast<phi::DenseTensor>(var.impl())->numel();
    } else {
      VLOG(3) << "var " << var.name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }

    group_info.first.push_back(tensor_real_index);
    group_info.second += experimental::SizeOf(var_dtype) * var_size;
    // group_info.second += framework::SizeOfType(var_dtype) * var_size;

    if (group_limit_index.find(var_dtype) == group_limit_index.end()) {
      // means it is the first var of var_dtype
      group_limit_index[var_dtype] = 0;
    }
    auto &cur_limit_index = group_limit_index[var_dtype];
    if (group_info.second >= group_size_limits[cur_limit_index]) {
      // exceed group capacity and create a new group
      res.emplace_back(std::move(group_info.first));
      group_info = std::pair<std::vector<size_t>, size_t>();
      cur_limit_index =
          (std::min)(cur_limit_index + 1, group_size_limits.size() - 1);
    }
  }

  // add the final groups
  for (auto &e : next_group) {
    auto &group_info = e.second;
    if (!group_info.first.empty()) {
      res.emplace_back(std::move(group_info.first));
    }
  }

  for (const auto &group_index : res) {
    PADDLE_ENFORCE_NE(
        group_index.empty(), true,
        platform::errors::PreconditionNotMet(
            "AssignGroupBySize construct empty group, please check."));
  }
  if (tensor_indices.empty()) {
    std::sort(res.begin(), res.end(),
              [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
                return x.front() < y.front();
              });
  }
  return res;
}

}  //  namespace distributed
}  //  namespace paddle
