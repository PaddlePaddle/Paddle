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

#include "paddle/fluid/framework/tensor.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#endif

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_CUDA
class CUDAGraphWithInOuts {
 public:
  template <typename Callable>
  CUDAGraphWithInOuts(Callable &&callable,
                      platform::CUDAPlace place,
                      const std::vector<const phi::DenseTensor *> &in_ptrs,
                      cudaStreamCaptureMode mode,
                      int64_t pool_id) {
    in_indices_.resize(in_ptrs.size());
    ins_.reserve(in_ptrs.size());
    int64_t valid_in_idx = 0;
    for (size_t i = 0; i < in_ptrs.size(); ++i) {
      if (in_ptrs[i] == nullptr) {
        in_indices_[i] = -1;
      } else {
        in_indices_[i] = (valid_in_idx++);
        ins_.push_back(*in_ptrs[i]);
      }
    }

    platform::BeginCUDAGraphCapture(place, mode, pool_id);
    auto out_ptrs = callable(in_ptrs);
    graph_ = platform::EndCUDAGraphCapture();
    graph_->Replay();

    out_indices_.resize(out_ptrs.size());
    outs_.reserve(out_ptrs.size());
    int64_t valid_out_idx = 0;
    for (size_t i = 0; i < out_ptrs.size(); ++i) {
      if (out_ptrs[i] == nullptr) {
        out_indices_[i] = -1;
      } else {
        out_indices_[i] = (valid_out_idx++);
        outs_.push_back(*out_ptrs[i]);
      }
    }
  }

  void Run(const std::vector<const phi::DenseTensor *> &ins) {
    PADDLE_ENFORCE_EQ(
        ins.size(),
        in_indices_.size(),
        phi::errors::InvalidArgument("The input number does not match."));
    for (size_t i = 0; i < in_indices_.size(); ++i) {
      if (in_indices_[i] >= 0) {
        auto *dst = &ins_[in_indices_[i]];
        framework::TensorCopy(*ins[i], dst->place(), dst);
      }
    }
    graph_->Replay();
  }

  std::vector<phi::DenseTensor *> GetOutputs() {
    std::vector<phi::DenseTensor *> outs(out_indices_.size());
    for (size_t i = 0; i < out_indices_.size(); ++i) {
      if (out_indices_[i] >= 0) {
        outs[i] = &outs_[out_indices_[i]];
      }
    }
    return outs;
  }

  int64_t PoolID() const { return graph_->PoolID(); }

 private:
  std::unique_ptr<platform::CUDAGraph> graph_;
  std::vector<phi::DenseTensor> ins_;
  std::vector<phi::DenseTensor> outs_;
  std::vector<int64_t> in_indices_;
  std::vector<int64_t> out_indices_;
};

template <typename Callable>
static std::unique_ptr<CUDAGraphWithInOuts> CaptureCUDAGraph(
    Callable &&callable,
    const framework::ExecutionContext &ctx,
    const std::vector<std::string> &input_names,
    const std::vector<std::string> &output_names,
    cudaStreamCaptureMode mode,
    int64_t pool_id) {
  std::vector<const phi::DenseTensor *> inputs;
  for (const auto &name : input_names) {
    auto input_tensors = ctx.MultiInput<phi::DenseTensor>(name);
    inputs.insert(inputs.end(), input_tensors.begin(), input_tensors.end());
  }

  auto func = [&](const std::vector<const phi::DenseTensor *> &inputs) {
    callable(ctx);
    std::vector<phi::DenseTensor *> outputs;
    for (const auto &name : output_names) {
      auto output_tensors = ctx.MultiOutput<phi::DenseTensor>(name);
      outputs.insert(
          outputs.end(), output_tensors.begin(), output_tensors.end());
    }
    return outputs;
  };

  return std::make_unique<CUDAGraphWithInOuts>(
      func, ctx.GetPlace(), inputs, mode, pool_id);
}

static void ExecuteCUDAGraph(const framework::ExecutionContext &ctx,
                             const std::vector<std::string> &input_names,
                             const std::vector<std::string> &output_names,
                             CUDAGraphWithInOuts *graph) {
  std::vector<const phi::DenseTensor *> inputs;
  for (const auto &name : input_names) {
    auto input_tensors = ctx.MultiInput<phi::DenseTensor>(name);
    inputs.insert(inputs.end(), input_tensors.begin(), input_tensors.end());
  }

  graph->Run(inputs);
  auto outputs = graph->GetOutputs();

  size_t idx = 0;
  for (const auto &name : output_names) {
    auto output_tensors = ctx.MultiOutput<phi::DenseTensor>(name);
    for (auto *out_t : output_tensors) {
      if (outputs[idx] != nullptr) {
        *out_t = *outputs[idx];
      } else {
        PADDLE_ENFORCE_EQ(
            out_t,
            nullptr,
            phi::errors::InvalidArgument(
                "The %d-th output variable should be nullptr.", idx));
      }
      ++idx;
    }
  }
}
#else
class CUDAGraphWithInOuts {};
#endif

}  // namespace operators
}  // namespace paddle
