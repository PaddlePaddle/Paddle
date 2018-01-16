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

#include <string.h>
#include "paddle/framework/op_registry.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class CTCGreedyDecodeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* output = ctx.Output<LoDTensor>("Output");
    const size_t level = 0;
    auto input_lod = framework::ToAbsOffset(input->lod());

    // check input dims and lod
    auto input_dims = input->dims();
    PADDLE_ENFORCE_EQ(input_dims[0],
                      static_cast<int64_t>(input_lod[level].back()),
                      "The first dimension of Input(Input) should be equal to "
                      "the sum of all sequences' lengths.");

    const size_t num_sequences = input_lod[level].size() - 1;
    size_t blank = static_cast<size_t>(ctx.Attr<int>("blank"));
    bool merge_repeated = ctx.Attr<bool>("merge_repeated");

    // merge repeated tokens and delete blank
    std::vector<std::vector<int>> pathes(num_sequences);
    std::vector<size_t> output_lod0(1, 0);
    const T* input_data = input->data<T>();
    for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      T prev_token = -1;
      for (size_t i = input_lod[level][seq_idx];
           i < input_lod[level][seq_idx + 1]; ++i) {
        if (input_data[i] != blank &&
            !(merge_repeated && input_data[i] == prev_token)) {
          pathes[seq_idx].push_back(input_data[i]);
        }
        prev_token = input_data[i];
      }
      output_lod0.push_back(output_lod0.back() + pathes[seq_idx].size());
    }

    // set output lod
    framework::LoD output_lod;
    output_lod.push_back(output_lod0);
    output->set_lod(output_lod);

    // resize output dims
    T* output_data = output->mutable_data<T>(
        {static_cast<int64_t>(output_lod0.back()), 1}, ctx.GetPlace());

    // copy result to output
    for (int i = 0; i < num_sequences; ++i) {
      memcpy(output_data + output_lod0[i], pathes[i].data(),
             sizeof(int) * pathes[i].size());
    }
  }
};

}  // namespace operators
}  // namespace paddle
