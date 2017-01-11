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

#include <gtest/gtest.h>
#include "FunctionTest.h"
#include "paddle/math/Matrix.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

void testMatrixProjectionForward(int context_start,
                                 size_t context_length,
                                 bool is_padding,
                                 size_t batch_size,
                                 size_t input_dim) {
  size_t pad = std::max(0, -context_start) +
               std::max(0, (int)(context_start + context_length - 1));
  if (pad == 0) is_padding = false;

  FunctionCompare compare("ContextProjectionForward",
                          FuncConfig()
                              .set("context_length", context_length)
                              .set("context_start", context_start)
                              .set("begin_pad", std::max(0, -context_start)));

  CpuMatrix cpu_in(batch_size, input_dim);
  cpu_in.randomizeUniform();
  GpuMatrix gpu_in(batch_size, input_dim);
  gpu_in.copyFrom(cpu_in);
  auto cpu_weight =
      is_padding ? std::make_shared<CpuMatrix>(pad, input_dim) : nullptr;
  auto gpu_weight =
      is_padding ? std::make_shared<GpuMatrix>(pad, input_dim) : nullptr;
  if (is_padding) {
    cpu_weight->randomizeUniform();
    gpu_weight->copyFrom(*cpu_weight);
  }
  IVectorPtr cpu_seq;
  generateSequenceStartPositions(batch_size, cpu_seq);
  IVectorPtr gpu_seq = IVector::create(cpu_seq->getSize(), true);
  gpu_seq->copyFrom(*cpu_seq);

  CpuMatrix cpu_out(batch_size, input_dim * context_length);
  GpuMatrix gpu_out(batch_size, input_dim * context_length);
  cpu_out.randomizeUniform();
  gpu_out.copyFrom(cpu_out);

  BufferArgs cpu_inputs;
  BufferArgs cpu_outputs;
  cpu_inputs.addArg(cpu_in);
  cpu_inputs.addArg(cpu_weight ? *cpu_weight
                               : CpuMatrix(nullptr, 0, input_dim));
  cpu_inputs.addArg(*cpu_seq);
  cpu_outputs.addArg(cpu_out, ADD_TO);

  compare.getCpuFunction()->calc(cpu_inputs, cpu_outputs);

  BufferArgs gpu_inputs;
  BufferArgs gpu_outputs;
  gpu_inputs.addArg(gpu_in);
  gpu_inputs.addArg(gpu_weight ? *gpu_weight
                               : GpuMatrix(nullptr, 0, input_dim));
  gpu_inputs.addArg(*gpu_seq);
  gpu_outputs.addArg(gpu_out, ADD_TO);

  compare.getGpuFunction()->calc(gpu_inputs, gpu_outputs);

  autotest::TensorCheckEqual(cpu_out, gpu_out);
}

void testMatrixProjectionBackward(int context_start,
                                  int context_length,
                                  bool is_padding,
                                  size_t batch_size,
                                  size_t input_dim) {
  size_t pad = std::max(0, -context_start) +
               std::max(0, (int)(context_start + context_length - 1));
  if (pad == 0) is_padding = false;

  FunctionCompare compare("ContextProjectionBackward",
                          FuncConfig()
                              .set("context_length", context_length)
                              .set("context_start", context_start)
                              .set("begin_pad", std::max(0, -context_start))
                              .set("is_padding", is_padding)
                              .set("total_pad", pad));

  CpuMatrix cpu_in_grad(batch_size, input_dim);
  cpu_in_grad.randomizeUniform();
  GpuMatrix gpu_in_grad(batch_size, input_dim);
  gpu_in_grad.copyFrom(cpu_in_grad);

  CpuMatrix cpu_out_grad(batch_size, input_dim * context_length);
  cpu_out_grad.randomizeUniform();
  GpuMatrix gpu_out_grad(batch_size, input_dim * context_length);
  gpu_out_grad.copyFrom(cpu_out_grad);

  IVectorPtr cpu_seq;
  generateSequenceStartPositions(batch_size, cpu_seq);
  IVectorPtr gpu_seq = IVector::create(cpu_seq->getSize(), true);
  gpu_seq->copyFrom(*cpu_seq);

  auto cpu_w_grad =
      is_padding ? std::make_shared<CpuMatrix>(pad, input_dim) : nullptr;
  auto gpu_w_grad =
      is_padding ? std::make_shared<GpuMatrix>(pad, input_dim) : nullptr;
  if (is_padding) {
    cpu_w_grad->randomizeUniform();
    gpu_w_grad->copyFrom(*cpu_w_grad);
  }

  BufferArgs cpu_inputs;
  BufferArgs cpu_outputs;
  cpu_inputs.addArg(cpu_out_grad, *cpu_seq);
  cpu_outputs.addArg(cpu_in_grad, ADD_TO);
  cpu_outputs.addArg(
      cpu_w_grad ? *cpu_w_grad : CpuMatrix(nullptr, 0, input_dim), ADD_TO);

  compare.getCpuFunction()->calc(cpu_inputs, cpu_outputs);

  BufferArgs gpu_inputs;
  BufferArgs gpu_outputs;
  gpu_inputs.addArg(gpu_out_grad, *gpu_seq);
  gpu_outputs.addArg(gpu_in_grad, ADD_TO);
  gpu_outputs.addArg(
      gpu_w_grad ? *gpu_w_grad : GpuMatrix(nullptr, 0, input_dim), ADD_TO);

  compare.getGpuFunction()->calc(gpu_inputs, gpu_outputs);

  autotest::TensorCheckErr(cpu_in_grad, gpu_in_grad);
  if (is_padding) {
    autotest::TensorCheckErr(*cpu_w_grad, *gpu_w_grad);
  }
}

TEST(ContextProjection, projection) {
  for (auto context_start : {-5, -3, -1, 0, 3}) {
    for (auto context_length : {1, 2, 5, 7}) {
      for (auto trainable_padding : {false, true}) {
        for (auto batch_size : {1, 2, 5, 20, 100}) {
          for (auto input_dim : {15, 32, 63, 128, 200}) {
            VLOG(3) << " context_start=" << context_start
                    << " context_length=" << context_length
                    << " trainable_padding=" << trainable_padding
                    << " batch_size=" << batch_size
                    << " input_dim=" << input_dim;
            testMatrixProjectionForward(context_start,
                                        context_length,
                                        trainable_padding,
                                        batch_size,
                                        input_dim);
            testMatrixProjectionBackward(context_start,
                                         context_length,
                                         trainable_padding,
                                         batch_size,
                                         input_dim);
          }
        }
      }
    }
  }
}
