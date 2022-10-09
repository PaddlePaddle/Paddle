/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;
static constexpr int TopKPosPaddingId = -1;

namespace details {

template <typename T>
static void get_topk_pos(const T* data, int length, int k, int* pos) {
  VLOG(3) << "length: " << length << " , k : " << k;

  std::priority_queue<std::pair<T, int>,
                      std::vector<std::pair<T, int>>,
                      std::greater<std::pair<T, int>>>
      topk_queue;

  for (int i = 0; i < length; ++i) {
    T elem = data[i];
    if (topk_queue.size() < static_cast<size_t>(k)) {
      topk_queue.emplace(elem, i);
    } else {
      if (elem >= topk_queue.top().first) {
        // replace top node if found a bigger value
        topk_queue.pop();
        topk_queue.emplace(elem, i);
      }
    }
  }
  // reversely assign value
  int real_k = topk_queue.size();
  for (int i = real_k - 1; i >= 0; --i) {
    pos[i] = topk_queue.top().second;
    topk_queue.pop();
  }
  // if length of data is less than k, fill TopKPosPaddingId at the end of pos.
  for (int i = real_k; i < k; ++i) {
    pos[i] = TopKPosPaddingId;
  }
}
}  // namespace details

template <typename DeviceContext, typename T>
class SequenceTopkAvgPoolingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* row = context.Input<LoDTensor>("ROW");
    auto* col = context.Input<LoDTensor>("COLUMN");
    auto* out = context.Output<LoDTensor>("Out");
    auto* pos = context.Output<phi::DenseTensor>("pos");

    PADDLE_ENFORCE_EQ(
        in->lod().empty(),
        false,
        platform::errors::InvalidArgument(
            "Input(X) Tensor of SequenceTopkAvgPoolingOp does not "
            "contain LoD information."));
    PADDLE_ENFORCE_EQ(
        row->lod().empty(),
        false,
        platform::errors::InvalidArgument(
            "Input(ROW) Tensor of SequenceTopkAvgPoolingOp does not "
            "contain LoD information."));
    PADDLE_ENFORCE_EQ(
        col->lod().empty(),
        false,
        platform::errors::InvalidArgument(
            "Input(COLUMN) Tensor of SequenceTopkAvgPoolingOp does "
            "not contain LoD information."));

    auto channel_num = context.Attr<int>("channel_num");
    auto topks = context.Attr<std::vector<int>>("topks");
    auto k_num = topks.size();
    auto max_k = topks[topks.size() - 1];
    PADDLE_ENFORCE_GE(max_k,
                      0,
                      platform::errors::InvalidArgument(
                          "Expected max_k >= 0, but received %d.", max_k));
    std::vector<int> vec_pos_shape;
    auto in_lod = in->lod()[0];

    auto row_lod = row->lod()[0];
    auto col_lod = col->lod()[0];
    int batch_size = row_lod.size() - 1;
    int pos_total_size = row_lod[batch_size] * channel_num * max_k;
    vec_pos_shape.push_back(pos_total_size);
    pos->Resize({phi::make_ddim(vec_pos_shape)});
    auto pos_data = pos->mutable_data<int>(context.GetPlace());

    int offset = 0;
    framework::Vector<size_t> vec_out_lod;
    vec_out_lod.reserve(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
      offset = row_lod[i];
      vec_out_lod.push_back(offset);
    }

    framework::LoD lod_temp;
    lod_temp.push_back(vec_out_lod);
    out->set_lod(lod_temp);

    auto din_data = in->data<T>();
    auto dout_data = out->mutable_data<T>(context.GetPlace());

    T* sum_data = new T[max_k];
    for (int i = 0; i < batch_size; ++i) {
      int total_size = in_lod[i + 1] - in_lod[i];
      int row_size = row_lod[i + 1] - row_lod[i];
      int col_size = col_lod[i + 1] - col_lod[i];
      PADDLE_ENFORCE_EQ(total_size,
                        channel_num * row_size * col_size,
                        platform::errors::PreconditionNotMet(
                            "Expected total_size == channel_num * row_size * "
                            "col_size, but got %d != %d.",
                            total_size,
                            channel_num * row_size * col_size));

      int feature_num = row_size * col_size;
      for (int j = 0; j < channel_num; ++j) {
        auto input_offset_feature_data = din_data + in_lod[i] + j * feature_num;

        for (int r = 0; r < row_size; ++r) {
          auto row_data = input_offset_feature_data + r * col_size;

          auto pos_slice_data = pos_data + row_lod[i] * channel_num * max_k +
                                r * channel_num * max_k + j * max_k;
          auto out_slice_data = dout_data + row_lod[i] * channel_num * k_num +
                                r * channel_num * k_num + j * k_num;

          details::get_topk_pos<T>(row_data, col_size, max_k, pos_slice_data);
          if (pos_slice_data[0] == TopKPosPaddingId) {
            sum_data[0] = 0.0;
          } else {
            sum_data[0] = row_data[pos_slice_data[0]];
          }
          for (int k = 1; k < max_k; ++k) {
            if (pos_slice_data[k] == TopKPosPaddingId) {
              sum_data[k] = sum_data[k - 1];
            } else {
              sum_data[k] = sum_data[k - 1] + row_data[pos_slice_data[k]];
            }
          }
          for (size_t k = 0; k < k_num; ++k) {
            out_slice_data[k] = sum_data[topks[k] - 1] / topks[k];
          }
        }
      }
    }
    delete[] sum_data;
  }
};

template <typename DeviceContext, typename T>
class SequenceTopkAvgPoolingGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_in = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* pos_input = context.Input<phi::DenseTensor>("pos");
    auto* row_input = context.Input<LoDTensor>("ROW");
    auto* col_input = context.Input<LoDTensor>("COLUMN");
    auto* forward_input = context.Input<LoDTensor>("X");

    int batch_size = row_input->lod()[0].size() - 1;
    auto channel_num = context.Attr<int>("channel_num");
    auto topks = context.Attr<std::vector<int>>("topks");
    auto k_num = topks.size();
    auto max_k = topks[k_num - 1];

    auto out_lod = forward_input->lod();
    d_in->set_lod(out_lod);

    d_in->mutable_data<T>(context.GetPlace());
    auto pos_data = pos_input->data<int>();
    auto dout_data = d_out->data<T>();

    auto& dev_ctx = context.template device_context<phi::CPUContext>();
    phi::funcs::SetConstant<phi::CPUContext, T> zero;
    zero(dev_ctx, d_in, static_cast<T>(0.0));

    auto din_data = d_in->data<T>();

    auto out_offset = out_lod[0];
    auto row_lod = row_input->lod()[0];
    auto col_lod = col_input->lod()[0];

    for (int i = 0; i < batch_size; ++i) {
      int row_size = row_lod[i + 1] - row_lod[i];
      int col_size = col_lod[i + 1] - col_lod[i];
      int feature_num = row_size * col_size;

      for (int j = 0; j < channel_num; ++j) {
        auto in_offset_feature_data =
            din_data + out_offset[i] + j * feature_num;

        for (int r = 0; r < row_size; r++) {
          auto row_data = dout_data + row_lod[i] * channel_num * k_num +
                          r * channel_num * k_num + j * k_num;
          auto pos_slice_data = pos_data + row_lod[i] * channel_num * max_k +
                                r * channel_num * max_k + j * max_k;
          auto in_slice_data = in_offset_feature_data + r * col_size;

          for (size_t m = 0; m < k_num; ++m) {
            for (int k = 0; k < topks[m]; ++k) {
              if (pos_slice_data[k] == TopKPosPaddingId) {
                break;
              } else {
                in_slice_data[pos_slice_data[k]] += row_data[m] / topks[m];
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
