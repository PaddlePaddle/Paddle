// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CudnnRnnFlattenWeightKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto weight = ctx.Input<framework::Tensor>("Weight");
    auto weight_ih = ctx.MultiOutput<framework::Tensor>("WeightIh");
    auto weight_hh = ctx.MultiOutput<framework::Tensor>("WeightHh");
    auto bias_ih = ctx.MultiOutput<framework::Tensor>("BiasIh");
    auto bias_hh = ctx.MultiOutput<framework::Tensor>("BiasHh");

    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int input_size = ctx.Attr<int>("input_size");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    int gate_size = 4 * hidden_size;
    int n_direct = is_bidirec ? 2 : 1;

    int offset = 0;
    int size = 0;
    for (int i = 0; i < num_layers; ++i) {
      for (int j = 0; j < n_direct; ++j) {
        int weight_num = i * n_direct + j;
        size = i == 0 ? gate_size * input_size : gate_size * hidden_size;
        weight_ih[weight_num]->ShareDataWith(*weight);
        weight_ih[weight_num]->set_offset(offset);
        weight_ih[weight_num]->Resize({size});
        offset += size * sizeof(T);

        size = gate_size * hidden_size;
        weight_hh[weight_num]->ShareDataWith(*weight);
        weight_hh[weight_num]->set_offset(offset);
        weight_hh[weight_num]->Resize({size});
        offset += size * sizeof(T);

        size = gate_size;
        bias_ih[weight_num]->ShareDataWith(*weight);
        bias_ih[weight_num]->set_offset(offset);
        bias_ih[weight_num]->Resize({size});
        offset += size * sizeof(T);

        size = gate_size;
        bias_hh[weight_num]->ShareDataWith(*weight);
        bias_hh[weight_num]->set_offset(offset);
        bias_hh[weight_num]->Resize({size});
        offset += size * sizeof(T);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
