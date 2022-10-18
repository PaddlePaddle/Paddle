/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sequence_pooling.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

class FusionSeqPoolCVMConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class FusionSeqPoolCVMConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T>
void CvmGradComputeKernel(const bool use_cvm, const int64_t item_width, const int64_t dy_offset,
                          const T& CVM, const T** DY, T** DX) {
  const auto cvm_offset = use_cvm ? 0 : 2;

  std::memcpy(*DX + cvm_offset, *DY, (item_width - cvm_offset) * sizeof(T));

  (*DX)[0] = (&CVM)[0];
  (*DX)[1] = (&CVM)[1];

  (*DX) += item_width;
  (*DY) += dy_offset;
  // (*DY) += item_width - cvm_offset;
}
class FusionSeqPoolCVMConcatGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class FusionSeqPoolCVMConcatGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dOut =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    const Tensor* cvm = context.Input<Tensor>("CVM");
    // const T* cvm_data = cvm->data<T>();

    auto dxs = context.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));

    auto use_cvm = context.Attr<bool>("use_cvm");
    std::string pooltype = context.Attr<std::string>("pooltype");
    math::SequencePoolGradFunctor<DeviceContext, T> pool;
    size_t n = dxs.size();

    auto cvm_offset = use_cvm ? 0 : 2;
    auto offset = 2;

    for (size_t k = 0; k < n; k++) {
      //cvm
      auto dx = dxs[k];
      auto batch_size = dx->dims()[0];
      auto item_size = dx->numel() / batch_size;
      auto dout_offset = dOut->dims()[1];
      const T* dout_data = dOut->data<T>() + k * (item_size-cvm_offset);

      int tmp_cvm_dx_bs = dOut->dims()[0];

      Tensor tmp_cvm_dx;
      T* tmp_cvm_dx_data = tmp_cvm_dx.mutable_data<T>({tmp_cvm_dx_bs, item_size}, context.GetPlace());

      const T* cvm_data = cvm->data<T>();
      // for Input X do not have Lod Information.
      // if (dx->NumLevels() == 0) {//TODO:
        for (int x = 0; x < tmp_cvm_dx_bs; ++x) {
          CvmGradComputeKernel(use_cvm, item_size, dout_offset, *cvm_data, &dout_data,
                              &tmp_cvm_dx_data);
          cvm_data += offset;
        }
      // } else {
      //   auto lod = dx->lod()[0];
      //   int seq_num = static_cast<int>(lod.size()) - 1;
      //   for (int i = 0; i < seq_num; ++i) {
      //     for (size_t j = 0; j < lod[i + 1] - lod[i]; ++j) {
      //       CvmGradComputeKernel(use_cvm, item_size, *cvm_data, dout_data,
      //                           tmp_cvm_dx_data);
      //     }
      //     cvm_data += offset;
      //   }
      // }
      //seqpool
      dx->mutable_data<T>(context.GetPlace());
      pool(context.template device_context<DeviceContext>(), pooltype, tmp_cvm_dx, dx);
    }
  }
};
}  // namespace operators
}  // namespace paddle
