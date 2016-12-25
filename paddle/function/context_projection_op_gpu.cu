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

#include "hl_base.h"
#include "context_projection_op.h"

namespace paddle {

template <bool padding>
__global__ void KeContextProjectionForward(const real* input,
                                           const int* sequence,
                                           const real* weight,
                                           real* output,
                                           int input_dim,
                                           int context_length,
                                           int context_start,
                                           int begin_pad) {
  int idx = threadIdx.x;
  int block_size = blockDim.x;
  int sequenceId = blockIdx.x;
  int seq_start = sequence[sequenceId];
  int seq_end = sequence[sequenceId+1];
  real value = 0;

  int instances = seq_end - seq_start + context_length - 1;
  output += seq_start * input_dim * context_length;
  input += seq_start * input_dim;
  for (int k = 0; k <= input_dim / block_size; k++) {
    if (idx < input_dim) {
      for (int i = 0; i < instances; i++) {
        // i + context_start;
        if ((i + context_start) < 0) {
          if (padding) {
            value = weight[i * input_dim + idx];
          } else {
            continue;
          }
        } else if ((i + context_start) >= (seq_end - seq_start)) {
          if (padding) {
            value =
              weight[(begin_pad + i + context_start - (seq_end - seq_start)) *
                         input_dim + idx];
          } else {
            continue;
          }
        } else {
          value = input[(i + context_start) * input_dim + idx];
        }

        int outx = (i - context_length) < 0 ? i : (context_length - 1);
        int outy = (i - context_length) < 0 ? 0 : (i - (context_length - 1));
        real* output_r =
          output + outy * input_dim * context_length + outx * input_dim;
        for (int j = outy; j < seq_end - seq_start; j++) {
          output_r[idx] += value;
          if (j - outy == outx) break;
          output_r += (context_length - 1) * input_dim;
        }
      }
    }
    idx += block_size;
  }
}

void hl_context_projection_forward(const real* input,
                                   const int* sequence,
                                   real* weight,
                                   real* output,
                                   int num_sequences,
                                   int input_dim,
                                   int context_length,
                                   int context_start,
                                   int begin_pad,
                                   bool is_padding) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(output);
  CHECK(!is_padding || weight);

  int block_size = 128;
  int blocks_x = num_sequences;
  int blocks_y = 1;
  dim3 threads(block_size, 1);
  dim3 grid(blocks_x, blocks_y);

  if (is_padding) {
    KeContextProjectionForward<true><<< grid, threads, 0, STREAM_DEFAULT >>>
      (input, sequence, weight, output, input_dim,
       context_length, context_start, begin_pad);
  } else  {
    KeContextProjectionForward<false><<< grid, threads, 0, STREAM_DEFAULT >>>
      (input, sequence, weight, output, input_dim,
       context_length, context_start, begin_pad);
  }
  CHECK_SYNC("hl_context_projection_forward failed");
}

template <>
void ContextProjectionForward<DEVICE_TYPE_GPU>(Tensor& output,
                                               const Tensor& input,
                                               const Tensor& weight,
                                               const Tensor& sequence,
                                               size_t context_length,
                                               int context_start,
                                               size_t begin_pad,
                                               bool is_padding) {
  CHECK(output.getData() && input.getData() && sequence.getData());
  CHECK_EQ(output.dims_.size(), 2);
  CHECK_EQ(input.dims_.size(), 2);
  CHECK_EQ(weight.dims_.size(), 2);
  CHECK_EQ(sequence.dims_.size(), 1);
  CHECK_EQ(output.dims_[1], input.dims_[1] * context_length);

  hl_context_projection_forward(input.getData(),
                                reinterpret_cast<int*>(sequence.getData()),
                                weight.getData(),
                                output.getData(),
                                sequence.dims_[0] - 1,
                                input.dims_[1],
                                context_length,
                                context_start,
                                begin_pad,
                                is_padding);
}

}  // namespace paddle
