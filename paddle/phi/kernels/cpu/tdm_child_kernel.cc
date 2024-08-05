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

#include <algorithm>
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T,
          typename Context,
          typename InfoT = int,
          typename OutT = int>
void TDMChildInner(const Context &dev_ctx,
                   const phi::DenseTensor &input,
                   const phi::DenseTensor &tree_info,
                   int child_nums,
                   phi::DenseTensor *child,
                   phi::DenseTensor *mask) {
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
        common::errors::InvalidArgument(
            "input id of OP(paddle.incubate.layers.tdm_child) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            node_nums,
            input_data[input_ids]));
    PADDLE_ENFORCE_LE(
        0,
        input_data[input_ids],
        common::errors::InvalidArgument(
            "input id of OP(paddle.incubate.layers.tdm_child) "
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
  auto *child_data = dev_ctx.template Alloc<OutT>(child);
  auto *leaf_mask_data = dev_ctx.template Alloc<OutT>(mask);

  memcpy(child_data, &child_vec[0], sizeof(OutT) * output_nums);
  memcpy(leaf_mask_data, &item_mask_vec[0], sizeof(OutT) * output_nums);
}

template <typename T, typename Context>
void TDMChildKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &tree_info,
                    int child_nums,
                    phi::DataType dtype,
                    phi::DenseTensor *child,
                    phi::DenseTensor *leaf_mask) {
  const auto &input_type = x.dtype();
  bool input_type_match =
      input_type == DataType::INT32 || input_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(input_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(X) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(input_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  const auto &info_type = tree_info.dtype();
  bool info_type_match =
      info_type == DataType::INT32 || info_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(
      info_type_match,
      true,
      common::errors::InvalidArgument(
          "Input(TreeInfo) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          DataTypeToString(info_type),
          DataTypeToString(DataType::INT32),
          DataTypeToString(DataType::INT64)));

  auto output_type = dtype;
  bool out_type_match =
      output_type == DataType::INT32 || output_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(out_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Output(Child) & Output(LeafMask) holds the wrong "
                        "type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(output_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  if (info_type == DataType::INT32 && output_type == DataType::INT32) {
    TDMChildInner<T, Context, int, int>(
        dev_ctx, x, tree_info, child_nums, child, leaf_mask);
  } else if (info_type == DataType::INT64 && output_type == DataType::INT32) {
    TDMChildInner<T, Context, int64_t, int>(
        dev_ctx, x, tree_info, child_nums, child, leaf_mask);
  } else if (info_type == DataType::INT32 && output_type == DataType::INT64) {
    TDMChildInner<T, Context, int, int64_t>(
        dev_ctx, x, tree_info, child_nums, child, leaf_mask);
  } else if (info_type == DataType::INT64 && output_type == DataType::INT64) {
    TDMChildInner<T, Context, int64_t, int64_t>(
        dev_ctx, x, tree_info, child_nums, child, leaf_mask);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(tdm_child,
                   CPU,
                   ALL_LAYOUT,
                   phi::TDMChildKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
