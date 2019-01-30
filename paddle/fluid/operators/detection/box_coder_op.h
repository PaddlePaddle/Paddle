/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

enum class BoxCodeType { kEncodeCenterSize = 0, kDecodeCenterSize = 1 };

inline BoxCodeType GetBoxCodeType(const std::string& type) {
  if (type == "encode_center_size") {
    return BoxCodeType::kEncodeCenterSize;
  } else if (type == "decode_center_size") {
    return BoxCodeType::kDecodeCenterSize;
  }
  PADDLE_THROW("Not support type %s.", type);
}

template <typename DeviceContext, typename T>
class BoxCoderKernel : public framework::OpKernel<T> {
 public:
  void EncodeCenterSize(const framework::Tensor* target_box,
                        const framework::Tensor* prior_box,
                        const framework::Tensor* prior_box_var,
                        const bool normalized,
                        const std::vector<float> variance, T* output) const {
    int64_t row = target_box->dims()[0];
    int64_t col = prior_box->dims()[0];
    int64_t len = prior_box->dims()[1];
    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    const T* prior_box_var_data = nullptr;
    if (prior_box_var) prior_box_var_data = prior_box_var->data<T>();

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        T prior_box_width = prior_box_data[j * len + 2] -
                            prior_box_data[j * len] + (normalized == false);
        T prior_box_height = prior_box_data[j * len + 3] -
                             prior_box_data[j * len + 1] +
                             (normalized == false);
        T prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
        T prior_box_center_y =
            prior_box_data[j * len + 1] + prior_box_height / 2;

        T target_box_center_x =
            (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
        T target_box_center_y =
            (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
        T target_box_width = target_box_data[i * len + 2] -
                             target_box_data[i * len] + (normalized == false);
        T target_box_height = target_box_data[i * len + 3] -
                              target_box_data[i * len + 1] +
                              (normalized == false);

        size_t offset = i * col * len + j * len;
        output[offset] =
            (target_box_center_x - prior_box_center_x) / prior_box_width;
        output[offset + 1] =
            (target_box_center_y - prior_box_center_y) / prior_box_height;
        output[offset + 2] =
            std::log(std::fabs(target_box_width / prior_box_width));
        output[offset + 3] =
            std::log(std::fabs(target_box_height / prior_box_height));
        if (prior_box_var) {
          int prior_var_offset = 0;
          if (prior_box_var->dims().size() == 2) {
            prior_var_offset = j * len;
          }
          output[offset] /= prior_box_var_data[prior_var_offset];
          output[offset + 1] /= prior_box_var_data[prior_var_offset + 1];
          output[offset + 2] /= prior_box_var_data[prior_var_offset + 2];
          output[offset + 3] /= prior_box_var_data[prior_var_offset + 3];
        } else if (!(variance.empty())) {
          for (int k = 0; k < 4; ++k) {
            output[offset + k] /= static_cast<T>(variance[k]);
          }
        }
      }
    }
  }
  void DecodeCenterSize(const framework::Tensor* target_box,
                        const framework::Tensor* prior_box,
                        const framework::Tensor* prior_box_var,
                        const bool normalized, const int axis,
                        const std::vector<float> variance, T* output) const {
    int64_t row = target_box->dims()[0];
    int64_t col = target_box->dims()[1];
    int64_t len = target_box->dims()[2];

    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    const T* prior_box_var_data = nullptr;
    if (prior_box_var) prior_box_var_data = prior_box_var->data<T>();
    int prior_box_offset = 0;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        size_t offset = i * col * len + j * len;
        if (axis == 0) {
          prior_box_offset = j * len;
        } else if (axis == 1) {
          prior_box_offset = i * len;
        }
        T prior_box_width = prior_box_data[prior_box_offset + 2] -
                            prior_box_data[prior_box_offset] +
                            (normalized == false);
        T prior_box_height = prior_box_data[prior_box_offset + 3] -
                             prior_box_data[prior_box_offset + 1] +
                             (normalized == false);
        T prior_box_center_x =
            prior_box_data[prior_box_offset] + prior_box_width / 2;
        T prior_box_center_y =
            prior_box_data[prior_box_offset + 1] + prior_box_height / 2;

        T target_box_center_x = 0, target_box_center_y = 0;
        T target_box_width = 0, target_box_height = 0;
        T box_var_x = T(1), box_var_y = T(1);
        T box_var_w = T(1), box_var_h = T(1);
        if (prior_box_var) {
          int prior_var_offset = 0;
          if (prior_box_var->dims().size() == 2) {
            if (axis == 0)
              prior_var_offset = j * len;
            else if (axis == 1)
              prior_var_offset = i * len;
          }
          box_var_x = prior_box_var_data[prior_var_offset];
          box_var_y = prior_box_var_data[prior_var_offset + 1];
          box_var_w = prior_box_var_data[prior_var_offset + 2];
          box_var_h = prior_box_var_data[prior_var_offset + 3];
        } else if (!(variance.empty())) {
          box_var_x = static_cast<T>(variance[0]);
          box_var_y = static_cast<T>(variance[1]);
          box_var_w = static_cast<T>(variance[2]);
          box_var_h = static_cast<T>(variance[3]);
        }
        target_box_center_x =
            box_var_x * target_box_data[offset] * prior_box_width +
            prior_box_center_x;
        target_box_center_y =
            box_var_y * target_box_data[offset + 1] * prior_box_height +
            prior_box_center_y;
        target_box_width =
            std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width;
        target_box_height = std::exp(box_var_h * target_box_data[offset + 3]) *
                            prior_box_height;

        output[offset] = target_box_center_x - target_box_width / 2;
        output[offset + 1] = target_box_center_y - target_box_height / 2;
        output[offset + 2] =
            target_box_center_x + target_box_width / 2 - (normalized == false);
        output[offset + 3] =
            target_box_center_y + target_box_height / 2 - (normalized == false);
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* prior_box = context.Input<framework::Tensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = context.Output<framework::Tensor>("OutputBox");
    std::vector<float> variance = context.Attr<std::vector<float>>("variance");
    const int axis = context.Attr<int>("axis");
    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(target_box->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }
    if (prior_box_var) {
      PADDLE_ENFORCE(variance.empty(),
                     "Input 'PriorBoxVar' and attribute 'variance' should not"
                     "be used at the same time.");
    }
    if (!(variance.empty())) {
      PADDLE_ENFORCE(static_cast<int>(variance.size()) == 4,
                     "Size of attribute 'variance' should be 4");
    }
    auto code_type = GetBoxCodeType(context.Attr<std::string>("code_type"));
    bool normalized = context.Attr<bool>("box_normalized");

    auto row = target_box->dims()[0];
    auto col = prior_box->dims()[0];
    if (code_type == BoxCodeType::kDecodeCenterSize) {
      col = target_box->dims()[1];
    }
    auto len = prior_box->dims()[1];

    output_box->mutable_data<T>({row, col, len}, context.GetPlace());

    T* output = output_box->data<T>();
    if (code_type == BoxCodeType::kEncodeCenterSize) {
      EncodeCenterSize(target_box, prior_box, prior_box_var, normalized,
                       variance, output);
    } else if (code_type == BoxCodeType::kDecodeCenterSize) {
      DecodeCenterSize(target_box, prior_box, prior_box_var, normalized, axis,
                       variance, output);
    }
  }
};

}  // namespace operators
}  // namespace paddle
