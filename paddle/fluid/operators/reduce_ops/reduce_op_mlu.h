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

#pragma once

#ifdef PADDLE_WITH_MLU
#include <string>
#include <vector>
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

template <typename T>
void MLUReduceOp(const framework::ExecutionContext& context,
                 std::string reduce_name) {
  PADDLE_ENFORCE_EQ(
      platform::is_mlu_place(context.GetPlace()), true,
      platform::errors::Unavailable("This kernel only runs on MLU."));
  auto* input = context.Input<Tensor>("X");
  auto* output = context.Output<Tensor>("Out");
  output->mutable_data<T>(context.GetPlace());

  bool reduce_all = context.Attr<bool>("reduce_all");
  auto dims = context.Attr<std::vector<int>>("dim");
  auto input_dims = phi::vectorize(input->dims());
  const auto& input_dim_size = input->dims().size();
  std::vector<int> reduce_dims;
  if (reduce_all) {
    for (size_t i = 0; i < input_dims.size(); i++) {
      reduce_dims.push_back(static_cast<int>(i));
    }
  } else {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        reduce_dims.push_back(dims[i] + input_dim_size);
      } else {
        reduce_dims.push_back(dims[i]);
      }
    }
  }

  MLUCnnlTensorDesc input_desc(*input, CNNL_LAYOUT_ARRAY,
                               ToCnnlDataType(input->dtype()));
  MLUCnnlTensorDesc output_desc(*output, CNNL_LAYOUT_ARRAY,
                                ToCnnlDataType(output->dtype()));

  cnnlReduceOp_t reduce_op = GetMLUCnnlReduceOp(reduce_name);
  MLUCnnlReduceDesc reduction_desc(reduce_dims, reduce_op, ToCnnlDataType<T>(),
                                   CNNL_NOT_PROPAGATE_NAN,
                                   CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

  MLUCnnl::Reduce(context, true /*need_workspace*/, reduction_desc.get(),
                  nullptr, input_desc.get(), GetBasePtr(input),
                  0 /*indices_size*/, nullptr, nullptr, output_desc.get(),
                  GetBasePtr(output));
}

}  // namespace operators
}  // namespace paddle
#endif
