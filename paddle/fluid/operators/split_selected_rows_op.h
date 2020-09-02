/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SplitSelectedRowsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::SelectedRows>("X");
    auto outs = ctx.MultiOutput<framework::SelectedRows>("Out");
    auto height_sections = ctx.Attr<std::vector<int64_t>>("height_sections");

    auto abs_sections = ToAbsoluteSection(height_sections);

    auto& x_rows = x->rows();
    auto height = x->height();
    std::vector<std::vector<int>> outs_rows_idx;
    std::vector<std::vector<int>> outs_dense_idx;

    outs_rows_idx.resize(outs.size());
    outs_dense_idx.resize(outs.size());

    auto row_numel = x->value().numel() / x->value().dims()[0];
    auto src = x->value().data<T>();

    // split rows index into output sparse vars
    for (size_t i = 0; i < x_rows.size(); ++i) {
      auto& id = x_rows[i];
      PADDLE_ENFORCE_LT(id, height,
                        platform::errors::OutOfRange(
                            "Each row_id in x.rows must be less than x.height. "
                            "But received x.rows[%d] = %d, x.height = %d",
                            i, id, height));
      int out_idx = GetSectionIndex(id, abs_sections);
      outs_rows_idx[out_idx].push_back(id);
      outs_dense_idx[out_idx].push_back(i);
    }
    auto place = ctx.GetPlace();

    for (size_t i = 0; i < outs_rows_idx.size(); ++i) {
      auto rows_idx = outs_rows_idx[i];
      outs[i]->set_height(height_sections[i]);
      auto dims = x->GetCompleteDims();
      dims[0] = rows_idx.size();
      outs[i]->mutable_value()->mutable_data<T>(dims, x->place());
      outs[i]->mutable_rows()->clear();
      if (rows_idx.size() > 0) {
        for (auto idx : rows_idx) {
          auto id_offset = idx - abs_sections[i];
          PADDLE_ENFORCE_LT(
              id_offset, height_sections[i],
              platform::errors::OutOfRange("Each row_id in out.rows must be "
                                           "less than out.height. But recived "
                                           "out.rows = [%d], out.height = [%d]",
                                           id_offset, height_sections[i]));
          outs[i]->mutable_rows()->push_back(id_offset);
        }
        auto dst = outs[i]->mutable_value()->mutable_data<T>(ctx.GetPlace());
        for (size_t j = 0; j < rows_idx.size(); j++) {
          if (platform::is_cpu_place(place)) {
            memory::Copy(
                platform::CPUPlace(), dst + j * row_numel, platform::CPUPlace(),
                src + outs_dense_idx[i][j] * row_numel, sizeof(T) * row_numel);
          } else {
#ifdef PADDLE_WITH_CUDA
            auto stream = ctx.cuda_device_context().stream();
            memory::Copy(platform::CUDAPlace(), dst + j * row_numel,
                         platform::CUDAPlace(),
                         src + outs_dense_idx[i][j] * row_numel,
                         sizeof(T) * row_numel, stream);
#else
            PADDLE_THROW(platform::errors::Unavailable(
                "Paddle is not compiled with CUDA. Cannot visit cuda device"));
#endif
          }
        }
      }
      PADDLE_ENFORCE_EQ(rows_idx.size(), outs[i]->rows().size(),
                        platform::errors::InvalidArgument(
                            "rows should has the same size with tensor dim 0. "
                            "But received rows = %d, tensor's dim[0] = %d.",
                            rows_idx.size(), outs[i]->rows().size()));
    }
  }
};

}  // namespace operators
}  // namespace paddle
