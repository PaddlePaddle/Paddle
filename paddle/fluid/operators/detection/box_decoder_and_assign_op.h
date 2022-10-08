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
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class BoxDecoderAndAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* prior_box = context.Input<framework::LoDTensor>("PriorBox");
    auto* prior_box_var = context.Input<phi::DenseTensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* box_score = context.Input<framework::LoDTensor>("BoxScore");
    auto* output_box = context.Output<phi::DenseTensor>("DecodeBox");
    auto* output_assign_box =
        context.Output<phi::DenseTensor>("OutputAssignBox");
    int roi_num = target_box->dims()[0];
    int class_num = box_score->dims()[1];
    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    auto* prior_box_var_data = prior_box_var->data<T>();
    auto* box_score_data = box_score->data<T>();
    output_box->mutable_data<T>({roi_num, class_num * 4}, context.GetPlace());
    output_assign_box->mutable_data<T>({roi_num, 4}, context.GetPlace());
    T* output_box_data = output_box->data<T>();
    T* output_assign_box_data = output_assign_box->data<T>();
    const T bbox_clip = static_cast<T>(context.Attr<float>("box_clip"));

    for (int i = 0; i < roi_num; ++i) {
      T prior_box_width = prior_box_data[i * 4 + 2] - prior_box_data[i * 4] + 1;
      T prior_box_height =
          prior_box_data[i * 4 + 3] - prior_box_data[i * 4 + 1] + 1;
      T prior_box_center_x = prior_box_data[i * 4] + prior_box_width / 2;
      T prior_box_center_y = prior_box_data[i * 4 + 1] + prior_box_height / 2;
      for (int j = 0; j < class_num; ++j) {
        int64_t offset = i * class_num * 4 + j * 4;
        T dw = std::min(prior_box_var_data[2] * target_box_data[offset + 2],
                        bbox_clip);
        T dh = std::min(prior_box_var_data[3] * target_box_data[offset + 3],
                        bbox_clip);
        T target_box_center_x = 0, target_box_center_y = 0;
        T target_box_width = 0, target_box_height = 0;
        target_box_center_x =
            prior_box_var_data[0] * target_box_data[offset] * prior_box_width +
            prior_box_center_x;
        target_box_center_y = prior_box_var_data[1] *
                                  target_box_data[offset + 1] *
                                  prior_box_height +
                              prior_box_center_y;
        target_box_width = std::exp(dw) * prior_box_width;
        target_box_height = std::exp(dh) * prior_box_height;

        output_box_data[offset] = target_box_center_x - target_box_width / 2;
        output_box_data[offset + 1] =
            target_box_center_y - target_box_height / 2;
        output_box_data[offset + 2] =
            target_box_center_x + target_box_width / 2 - 1;
        output_box_data[offset + 3] =
            target_box_center_y + target_box_height / 2 - 1;
      }

      T max_score = -1;
      int max_j = -1;
      for (int j = 0; j < class_num; ++j) {
        T score = box_score_data[i * class_num + j];
        if (score > max_score && j > 0) {
          max_score = score;
          max_j = j;
        }
      }

      if (max_j > 0) {
        for (int pno = 0; pno < 4; pno++) {
          output_assign_box_data[i * 4 + pno] =
              output_box_data[i * class_num * 4 + max_j * 4 + pno];
        }
      } else {
        for (int pno = 0; pno < 4; pno++) {
          output_assign_box_data[i * 4 + pno] = prior_box_data[i * 4 + pno];
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
