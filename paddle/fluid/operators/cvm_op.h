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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
void CvmComputeKernel(const bool use_cvm,
                      const int64_t item_width,
                      const T** X,
                      T** Y) {
  const auto cvm_offset = use_cvm ? 0 : 2;

  std::memcpy(*Y, *X + cvm_offset, (item_width - cvm_offset) * sizeof(T));

  if (use_cvm) {
    (*Y)[0] = log((*Y)[0] + 1);
    (*Y)[1] = log((*Y)[1] + 1) - (*Y)[0];
  }

  (*X) += item_width;
  (*Y) += item_width - cvm_offset;
}

template <typename T>
void CvmGradComputeKernel(const bool use_cvm,
                          const int64_t item_width,
                          const T& CVM,
                          const T** DY,
                          T** DX) {
  const auto cvm_offset = use_cvm ? 0 : 2;

  std::memcpy(*DX + cvm_offset, *DY, (item_width - cvm_offset) * sizeof(T));

  (*DX)[0] = (&CVM)[0];
  (*DX)[1] = (&CVM)[1];

  (*DX) += item_width;
  (*DY) += item_width - cvm_offset;
}

template <typename T>
class CVMOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<phi::DenseTensor>("X");
    const T* x_data = x->data<T>();

    auto batch_size = x->dims()[0];
    auto item_size = x->numel() / batch_size;
    auto use_cvm = context.Attr<bool>("use_cvm");

    auto* y = context.Output<phi::DenseTensor>("Y");
    T* y_data = y->mutable_data<T>(context.GetPlace());

    // for Input X do not have Lod Information.
    if (x->NumLevels() == 0) {
      if (use_cvm) {
        for (int i = 0; i < batch_size; i++) {
          int cursor = i * item_size;
          y_data[cursor] = log(x_data[cursor] + 1);
          y_data[cursor + 1] = log(x_data[cursor + 1] + 1) - y_data[cursor];
          for (int j = 2; j < item_size; j++) {
            y_data[cursor + j] = x_data[cursor + j];
          }
        }
      } else {
        for (int i = 0; i < batch_size; i++) {
          CvmComputeKernel(use_cvm, item_size, &x_data, &y_data);
        }
      }
    } else {
      auto lod = x->lod()[0];
      for (size_t i = 0; i < lod.size() - 1; ++i) {
        for (size_t j = 0; j < lod[i + 1] - lod[i]; ++j) {
          CvmComputeKernel(use_cvm, item_size, &x_data, &y_data);
        }
      }
    }
  }
};

template <typename T>
class CVMGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dx = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    const phi::DenseTensor* cvm = context.Input<phi::DenseTensor>("CVM");
    const T* cvm_data = cvm->data<T>();

    const auto* dOut =
        context.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const T* dout_data = dOut->data<T>();

    auto use_cvm = context.Attr<bool>("use_cvm");

    auto offset = 2;
    auto batch_size = dx->dims()[0];
    auto item_size = dx->numel() / batch_size;

    // for Input X do not have Lod Information.
    if (dx->NumLevels() == 0) {
      for (int x = 0; x < batch_size; ++x) {
        CvmGradComputeKernel(
            use_cvm, item_size, *cvm_data, &dout_data, &dx_data);
        cvm_data += offset;
      }
    } else {
      auto lod = dx->lod()[0];
      int seq_num = static_cast<int>(lod.size()) - 1;
      for (int i = 0; i < seq_num; ++i) {
        for (size_t j = 0; j < lod[i + 1] - lod[i]; ++j) {
          CvmGradComputeKernel(
              use_cvm, item_size, *cvm_data, &dout_data, &dx_data);
        }
        cvm_data += offset;
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
