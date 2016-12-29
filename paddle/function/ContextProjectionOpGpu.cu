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
#include "ContextProjectionOp.h"

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

__global__ void KeContextProjectionBackwardData(real* out_grad,
                                                const int* sequence,
                                                real* in_grad,
                                                int input_dim,
                                                int context_length,
                                                int context_start) {
  int idx = threadIdx.x;
  int block_size = blockDim.x;
  int sequenceId = blockIdx.x;
  int seq_start = sequence[sequenceId];
  int seq_end = sequence[sequenceId+1];
  real value = 0;

  int instances = seq_end - seq_start + context_length - 1;
  out_grad += seq_start * input_dim * context_length;
  in_grad += seq_start * input_dim;
  for (int k = 0; k <= input_dim / block_size; k++) {
    if (idx < input_dim) {
      for (int i = 0; i < instances; i++) {
        if ((i + context_start) < 0) {
          continue;
        } else if ((i + context_start) >= (seq_end - seq_start)) {
          continue;
        } else {
          // value = 0;
          value = in_grad[(i + context_start) * input_dim + idx];
        }

        int outx = (i - context_length) < 0 ? i : (context_length - 1);
        int outy = (i - context_length) < 0 ? 0 : (i - (context_length - 1));
        real* output_r =
          out_grad + outy * input_dim * context_length + outx * input_dim;
        for (int j = outy; j < seq_end - seq_start; j++) {
          value += output_r[idx];
          if (j - outy == outx) break;
          output_r += (context_length - 1) * input_dim;
        }
        in_grad[(i + context_start) * input_dim + idx] = value;
      }
    }
    idx += block_size;
  }
}

void hl_context_projection_backward_data(real* out_grad,
                                         const int* sequence,
                                         real* input_grad,
                                         int num_sequences,
                                         int input_dim,
                                         int context_length,
                                         int context_start) {
  CHECK_NOTNULL(out_grad);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(input_grad);

  int block_size = 128;
  int blocks_x = num_sequences;
  int blocks_y = 1;
  dim3 threads(block_size, 1);
  dim3 grid(blocks_x, blocks_y);
  KeContextProjectionBackwardData<<< grid, threads, 0, STREAM_DEFAULT >>>
    (out_grad, sequence, input_grad, input_dim, context_length, context_start);
  CHECK_SYNC("hl_context_projection_backward_data failed");
}

template <>
void ContextProjectionBackwardData<DEVICE_TYPE_GPU>(Tensor& out_grad,
                                               Tensor& in_grad,
                                               const Tensor& sequence,
                                               size_t context_length,
                                               int context_start) {
  CHECK(in_grad.getData() && out_grad.getData() && sequence.getData());
  CHECK_EQ(out_grad.dims_.size(), 2);
  CHECK_EQ(in_grad.dims_.size(), 2);
  CHECK_EQ(sequence.dims_.size(), 1);
  CHECK_EQ(out_grad.dims_[1], in_grad.dims_[1] * context_length);

  hl_context_projection_backward_data(out_grad.getData(),
                reinterpret_cast<int*>(sequence.getData()),
                                      in_grad.getData(),
                                      sequence.dims_[0] - 1,
                                      in_grad.dims_[1],
                                      context_length,
                                      context_start);
}

template<int THREADS_X, int THREADS_Y>
__global__ void KeContextProjectionBackwardWeight(real* out_grad,
                                                  const int* sequence,
                                                  real* w_grad,
                                                  int num_sequences,
                                                  int w_dim,
                                                  int context_length,
                                                  int context_start,
                                                  int begin_pad) {
  __shared__ real sum_s[THREADS_Y][THREADS_X];
  int pad_of_block = (w_dim + THREADS_X - 1) / THREADS_X;
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  int padId = blockIdx.x / pad_of_block;
  int weight_idx = idx + THREADS_X * (blockIdx.x % pad_of_block);
  int instanceId;
  real value = 0;
  real* output_r;

  sum_s[idy][idx] = 0.0f;
  if (weight_idx < w_dim) {
    for (int seqId = idy; seqId < num_sequences; seqId += THREADS_Y) {
      int seq_start = sequence[seqId];
      int seq_end = sequence[seqId+1];
      output_r = out_grad + seq_start * w_dim * context_length;

      if (context_start < 0) {
        if (padId + context_start < 0) {
          instanceId = padId;
        } else {
          // begin_pad > 0;
          instanceId = (padId - begin_pad) +
            (seq_end - seq_start) - context_start;
        }
      } else {
        if (padId + (seq_end - seq_start) < context_start) {
          continue;
        } else {
          // begin_pad == 0;
          instanceId = padId + (seq_end - seq_start) - context_start;
        }
      }

      int outx = (instanceId - context_length) < 0 ?
                 instanceId : (context_length - 1);
      int outy = (instanceId - context_length) < 0 ?
                 0 : (instanceId - (context_length - 1));
      output_r += outy * w_dim * context_length + outx * w_dim;
      for (int j = outy; j < seq_end - seq_start; j++) {
        value += output_r[weight_idx];
        if (j - outy == outx) break;
        output_r += (context_length - 1) * w_dim;
      }
    }
    sum_s[idy][idx] = value;
  }
  __syncthreads();

  for (int stride = THREADS_Y/2; stride > 0; stride = stride/2) {
    if (idy < stride) {
      sum_s[idy][idx] += sum_s[idy + stride][idx];
    }
    __syncthreads();
  }
  __syncthreads();

  if (weight_idx < w_dim) {
    if (idy == 0) {
      w_grad[padId * w_dim + weight_idx] += sum_s[0][idx];
    }
  }
}

void hl_context_projection_backward_weight(real* out_grad,
                                           const int* sequence,
                                           real* w_grad,
                                           int num_sequences,
                                           int w_dim,
                                           size_t total_pad,
                                           int context_length,
                                           int context_start,
                                           int begin_pad) {
  CHECK_NOTNULL(out_grad);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(w_grad);

  int threads_x = 32;
  int threads_y = 32;
  int blocks_x = total_pad * ((w_dim + threads_x - 1) / threads_x);
  dim3 threads(threads_x, threads_y);
  dim3 grid(blocks_x, 1);

  KeContextProjectionBackwardWeight<32, 32>
    <<< grid, threads, 0, STREAM_DEFAULT >>>
    (out_grad, sequence, w_grad, num_sequences, w_dim,
     context_length, context_start, begin_pad);
  CHECK_SYNC("hl_context_projection_backward_weight failed");
}

template <>
void ContextProjectionBackwardWeight<DEVICE_TYPE_GPU>(Tensor& out_grad,
                                                      Tensor& w_grad,
                                                      const Tensor& sequence,
                                                      size_t context_length,
                                                      int context_start,
                                                      size_t total_pad,
                                                      size_t begin_pad) {
  CHECK(w_grad.getData() && out_grad.getData() && sequence.getData());
  CHECK_EQ(out_grad.dims_.size(), 2);
  CHECK_EQ(w_grad.dims_.size(), 2);
  CHECK_EQ(sequence.dims_.size(), 1);
  CHECK_EQ(out_grad.dims_[1], w_grad.dims_[1] * context_length);

  hl_context_projection_backward_weight(out_grad.getData(),
                    reinterpret_cast<int*>(sequence.getData()),
                                        w_grad.getData(),
                                        sequence.dims_[0] - 1,
                                        w_grad.dims_[1],
                                        total_pad,
                                        context_length,
                                        context_start,
                                        begin_pad);
}

template <>
void ContextProjectionBackward<DEVICE_TYPE_GPU>(Tensor& out_grad,
                                               Tensor& in_grad,
                                               Tensor& w_grad,
                                               const Tensor& sequence,
                                               size_t context_length,
                                               int context_start,
                                               size_t begin_pad,
                                               bool is_padding,
                                               size_t total_pad) {
    if (in_grad.getData()) {
        ContextProjectionBackwardData<DEVICE_TYPE_GPU>(out_grad,
                in_grad,
                sequence,
                context_length,
                context_start);
    }
    if (is_padding && w_grad.getData()) {
        ContextProjectionBackwardWeight<DEVICE_TYPE_GPU>(out_grad,
                w_grad,
                sequence,
                context_length,
                context_start,
                total_pad,
                begin_pad);
  }
}

}  // namespace paddle
