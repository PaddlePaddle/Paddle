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
    auto ancestor_nums = ctx.Attr<int>("Ancestor_nums");
    auto child_nums = ctx.Attr<int>("Child_nums");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &tree_emb_tensor = tree_emb_var->Get<LoDTensor>();
    auto dims = tree_emb_tensor.dims();
    int node_nums = dims[0];
    int length = dims[1];

    int input_ids_num = input_tensor.numel();
    int batch_size = input_tensor.dims()[0];
    VLOG(1) << "TDM child : input numel -> " << input_ids_num;

    std::vector<T> child_vec{};
    std::vector<T> item_mask_vec{};
    auto *input_data = input_tensor.data<T>();
    auto *tree_emb_data = tree_emb_tensor.data<int>();

    // Tree_emb: node_id : item_id; layer_id; ancestor_id; child_id
    for (int input_ids = 0; input_ids < input_ids_num; ++input_ids) {
      // if input_data[input_ids]>node_nums return false
      PADDLE_ENFORCE_LT(
          input_data[input_ids], node_nums,
          "input id of OP(fluid.layers.tdm_child) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          node_nums, input_data[input_ids]);
      PADDLE_ENFORCE_LE(
          0, input_data[input_ids],
          "input id of OP(fluid.layers.tdm_child) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          node_nums, input_data[input_ids]);

      bool has_child =
          (input_data[input_ids] == 0 ||
           tree_emb_data[static_cast<int>(input_data[input_ids]) * length +
                         3] == 0)
              ? false
              : true;

      if (has_child) {
        for (int child_ids = 0; child_ids < child_nums; ++child_ids) {
          T child_id =
              tree_emb_data[static_cast<int>(input_data[input_ids]) * length +
                            3 + child_ids];
          child_vec.push_back(child_id);
          T child_is_item =
              tree_emb_data[static_cast<int>(child_id) * length] == 0 ? 0 : 1;
          item_mask_vec.push_back(child_is_item);
        }
      } else {
        for (int child_ids = 0; child_ids < child_nums; ++child_ids) {
          child_vec.push_back(0);
          item_mask_vec.push_back(0);
        }
      }
    }

    auto *child_var = ctx.OutputVar("Child");
    auto *item_mask_var = ctx.OutputVar("Item_mask");

    int output_nums = child_vec.size();
    auto ddim = framework::make_ddim({batch_size, ancestor_nums, child_nums});

    auto *child_tensor = child_var->GetMutable<framework::LoDTensor>();
    child_tensor->Resize(ddim);
    auto *item_mask_tensor = item_mask_var->GetMutable<framework::LoDTensor>();
    item_mask_tensor->Resize(ddim);

    auto *child_data = child_tensor->mutable_data<T>(ctx.GetPlace());
    auto *item_mask_data = item_mask_tensor->mutable_data<T>(ctx.GetPlace());

    memcpy(child_data, &child_vec[0], sizeof(T) * output_nums);
    memcpy(item_mask_data, &item_mask_vec[0], sizeof(T) * output_nums);
  }
};

}  // namespace operators
}  // namespace paddle
