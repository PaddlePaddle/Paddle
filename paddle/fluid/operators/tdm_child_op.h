/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
class TDMChildKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("Input");
    auto *tree_emb_var = ctx.InputVar("Tree_embedding");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &tree_emb_tensor = tree_emb_var->Get<LoDTensor>();
    auto dims = tree_emb_tensor.dims();
    int node_nums = dims[0];
    int length = dims[1];

    int input_ids_num = input_tensor.numel();
    VLOG(1) << "TDM child : input numel -> " << input_ids_num;

    std::vector<int64_t> child_vec{};
    std::vector<int64_t> item_mask_vec{};
    auto *input_data = input_tensor.data<int64_t>();
    auto *tree_emb_data = tree_emb_tensor.data<int64_t>();

    // Tree_emb: node_id : item_id; layer_id; ancestor_id; child_id
    for (int input_ids = 0; input_ids < input_ids_num; ++input_ids) {
      // if input_data[input_ids]>node_nums return false
      bool has_child =
          tree_emb_data[input_data[input_ids] * length + 3] == 0 ? false : true;
      if (has_child) {
        for (int child_ids = 3; child_ids < length; ++child_ids) {
          int64_t child_id =
              tree_emb_data[input_data[input_ids] * length + child_ids];
          if (child_id) {
            child_vec.push_back(
                tree_emb_data[input_data[input_ids] * length + child_ids]);
            bool child_is_item =
                tree_emb_data[child_id * length] == 0 ? false : true;
            item_mask_vec.push_back(static_cast<int64_t>(child_is_item));
          }
        }
      }
    }

    auto *child_var = ctx.OutputVar("Child");
    auto *item_mask_var = ctx.OutputVar("Item_mask");

    int output_nums = child_vec.size();
    auto ddim = framework::make_ddim({output_nums, 1});

    auto *child_tensor = child_var->GetMutable<framework::LoDTensor>();
    child_tensor->Resize(ddim);
    auto *item_mask_tensor = item_mask_var->GetMutable<framework::LoDTensor>();
    item_mask_tensor->Resize(ddim);

    auto *child_data = child_tensor->mutable_data<int64_t>(context.GetPlace());
    auto *item_mask_data =
        item_mask_tensor->mutable_data<int64_t>(context.GetPlace());

    memcpy(child_data, &child_vec[0], sizeof(int64_t) * output_nums);
    memcpy(item_mask_data, &item_mask_vec[0], sizeof(int64_t) * output_nums);
  }
};

}  // namespace operators
}  // namespace paddle
