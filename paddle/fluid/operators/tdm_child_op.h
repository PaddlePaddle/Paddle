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

#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;
using LoD = framework::LoD;

template <typename T, typename InfoT = int, typename OutT = int>
void TDMChildInner(const framework::ExecutionContext &context,
                   const phi::DenseTensor &input,
                   const phi::DenseTensor &tree_info,
                   phi::DenseTensor *child,
                   phi::DenseTensor *mask) {
  auto child_nums = context.Attr<int>("child_nums");
  auto info_dims = tree_info.dims();
  int node_nums = info_dims[0];
  int length = info_dims[1];

  int input_ids_num = input.numel();
  VLOG(4) << "TDM child op: input numel ->  " << input_ids_num;

  std::vector<OutT> child_vec{};
  std::vector<OutT> item_mask_vec{};

  auto *input_data = input.data<T>();
  auto *tree_info_data = tree_info.data<InfoT>();

  // TreeInfo: node_id : item_id; layer_id; ancestor_id; child_id
  for (int input_ids = 0; input_ids < input_ids_num; ++input_ids) {
    PADDLE_ENFORCE_LT(
        input_data[input_ids],
        node_nums,
        platform::errors::InvalidArgument(
            "input id of OP(fluid.contrib.layers.tdm_child) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            node_nums,
            input_data[input_ids]));
    PADDLE_ENFORCE_LE(
        0,
        input_data[input_ids],
        platform::errors::InvalidArgument(
            "input id of OP(fluid.contrib.layers.tdm_child) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            node_nums,
            input_data[input_ids]));

    bool has_child =
        (input_data[input_ids] == 0 ||
         tree_info_data[static_cast<int>(input_data[input_ids]) * length + 3] ==
             0)
            ? false
            : true;

    if (has_child) {
      for (int child_ids = 0; child_ids < child_nums; ++child_ids) {
        OutT child_id = static_cast<OutT>(
            tree_info_data[static_cast<int>(input_data[input_ids]) * length +
                           3 + child_ids]);
        child_vec.push_back(child_id);
        OutT child_is_item = static_cast<OutT>(
            tree_info_data[static_cast<int>(child_id) * length] == 0 ? 0 : 1);
        item_mask_vec.push_back(child_is_item);
      }
    } else {
      for (int child_ids = 0; child_ids < child_nums; ++child_ids) {
        child_vec.push_back(0);
        item_mask_vec.push_back(0);
      }
    }
  }

  int output_nums = child_vec.size();
  auto *child_data = child->mutable_data<OutT>(context.GetPlace());
  auto *leaf_mask_data = mask->mutable_data<OutT>(context.GetPlace());

  memcpy(child_data, &child_vec[0], sizeof(OutT) * output_nums);
  memcpy(leaf_mask_data, &item_mask_vec[0], sizeof(OutT) * output_nums);
}

template <typename DeviceContext, typename T>
class TDMChildKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *tree_info_var = ctx.InputVar("TreeInfo");

    auto &input_tensor = input_var->Get<phi::DenseTensor>();
    const auto &input_type =
        framework::TransToProtoVarType(input_tensor.dtype());
    bool input_type_match = input_type == framework::proto::VarType::INT32 ||
                            input_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(input_type_match,
                      true,
                      platform::errors::InvalidArgument(
                          "Input(X) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(input_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto &tree_info_tensor = tree_info_var->Get<phi::DenseTensor>();
    const auto &info_type =
        framework::TransToProtoVarType(tree_info_tensor.dtype());
    bool info_type_match = info_type == framework::proto::VarType::INT32 ||
                           info_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        info_type_match,
        true,
        platform::errors::InvalidArgument(
            "Input(TreeInfo) holds the wrong type, it holds %s, but "
            "desires to be %s or %s",
            paddle::framework::DataTypeToString(info_type),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64)));

    auto *child_var = ctx.OutputVar("Child");
    auto *leaf_mask_var = ctx.OutputVar("LeafMask");
    auto *child_tensor = child_var->GetMutable<phi::DenseTensor>();
    auto *leaf_mask_tensor = leaf_mask_var->GetMutable<phi::DenseTensor>();

    auto output_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    bool out_type_match = output_type == framework::proto::VarType::INT32 ||
                          output_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(out_type_match,
                      true,
                      platform::errors::InvalidArgument(
                          "Output(Child) & Output(LeafMask) holds the wrong "
                          "type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(output_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    if (info_type == framework::proto::VarType::INT32 &&
        output_type == framework::proto::VarType::INT32) {
      TDMChildInner<T, int, int>(
          ctx, input_tensor, tree_info_tensor, child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT32) {
      TDMChildInner<T, int64_t, int>(
          ctx, input_tensor, tree_info_tensor, child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT32 &&
               output_type == framework::proto::VarType::INT64) {
      TDMChildInner<T, int, int64_t>(
          ctx, input_tensor, tree_info_tensor, child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT64) {
      TDMChildInner<T, int64_t, int64_t>(
          ctx, input_tensor, tree_info_tensor, child_tensor, leaf_mask_tensor);
    }
  }
};
}  // namespace operators
}  // namespace paddle
