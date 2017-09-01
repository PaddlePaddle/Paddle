/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "gather.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "scatter.h"

namespace paddle {
namespace operators {

using LODTensor = framework::LODTensor;

template <typename Place, typename T>
class ScatterOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *Ref = ctx.Input<LODTensor>("Ref");
    auto *Index = ctx.Input<LODTensor>("Index");
    auto *Updates = ctx.Input<LODTensor>("Updates");
    auto *Out = ctx.Output<LODTensor>("Out");

    // In place output: Out = Ref, Out[Index] += Updates
    Out->ShareDataWith<T>(*Ref);
    // Apply ScatterUpdate: Out[index] += Updates[:]
    ScatterUpdate<T>(ctx.GetPlace(), Updates, Index, Out);
  }
};

template <typename Place, typename T>
class ScatterGradientOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dRef = ctx.Output<LODTensor>(framework::GradVarName("Ref"));
    auto *dUpdates = ctx.Output<LODTensor>(framework::GradVarName("Updates"));
    auto *Index = ctx.Input<LODTensor>("Index");
    auto *dOut = ctx.Input<LODTensor>(framework::GradVarName("Out"));

    // In place gradient: dRef = dO
    dRef->ShareDataWith<T>(*dOut);
    dUpdates->mutable_data<T>(ctx.GetPlace());
    // Gradient by Gather: dUpdates += dO[Index]
    Gather<T>(ctx.GetPlace(), dOut, Index, dUpdates);
  }
};

}  // namespace operators
}  // namespace paddle
