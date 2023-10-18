/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/im2col.h"

namespace paddle {
namespace operators {

namespace math {

/*
 * \brief Context projection concatenates features in adjacent time-steps in
 * a sequence. The i-th row of the output is the concatenation of
 * context_length rows of the input. The context_length rows are the
 * consecutive rows from the i+shift_start row.
 * ContextProjectGradFunctor is the inverse process of ContextProjectFunctor.
 *
 * \param in            Input data.
 * \param Shape         The shape of Input data:
 *                        [mini-batch, input_hidden_size].
 *
 * \param padding_data  Padding data.
 * \param Shape         The shape of Padding data:
 *                        [up_pad + down_pad, input_hidden_size].
 *
 * \param col           Col data.
 * \param Shape         The shape of Col data:
 *                        [mini-batch, context_length * input_hidden_size].
 *
 * For a mini-batch of 2 variable lengths sentences, containing 3, and 1
 * time-steps:
 *
 * Assumed input (X) is a [4, M, N] float phi::DenseTensor, and X->lod()[0] =
 * [0, 3, 4]. Besides, for the sake of simplicity, we assume M=1 and N=2.
 *
 * X = [[a1, a2;
 *       b1, b2;
 *       c1, c2]
 *      [d1, d2]]
 *
 * This is to say that input (X) has 4 words and the dimension of each word
 * representation is 2.
 *
 * - Case1:
 *   If context_start is -1 and padding_trainable is false, we use zero to pad
 *   instead of learned weight to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out =[[0,  0,  a1, a2, b1, b2;
 *          a1, a2, b1, b2, c1, c2;
 *          b1, b2, c1, c2, 0,  0 ]
 *          [0,  0, d1, d2, 0,  0 ]]
 *
 * - Case2:
 *   If context_start is -1 and padding_trainable is true, we use learned weight
 *   to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out = [[w1, w2, a1, a2, b1, b2;
 *           a1, a2, b1, b2, c1, c2;
 *           b1, b2, c1, c2, w3, w4]
 *          [w1, w2, d1, d2, w3, w4]]
 *
 */

template <typename DeviceContext, typename T>
class ContextProjectFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& in,
                  const phi::DenseTensor* padding_data,
                  bool padding_trainable,
                  const int context_start,
                  const int context_length,
                  const int context_stride,
                  const int up_pad,
                  const int down_pad,
                  phi::DenseTensor* col) {
    auto lod_level_0 = in.lod()[0];

    phi::funcs::
        Im2ColFuseFunctor<phi::funcs::ColFormat::kOCF, DeviceContext, float>
            im2col_ocf_fuse;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    int sequence_width;
    sequence_width = in.dims()[1];

    int concurrency_size = 5;
    int thread_size = static_cast<int>(lod_level_0.size()) - 1;
    std::vector<std::thread> threads_vec(concurrency_size);

    int im_channels = 1;
    std::vector<int> im_height(thread_size, 0);
    int im_width = sequence_width;
    int filter_height = context_length;
    int filter_width = sequence_width;
    int col_width = 1;
    std::vector<int> col_height(thread_size, 0);
    // framework::Vector<size_t> lod_level_0_cuda = lod_level_0;
    // im_data, col_data
    std::vector<float*> im_datas(thread_size);
    std::vector<float*> col_datas(thread_size);
    int max_col_height = -1;

    int avg_ele = thread_size / concurrency_size;
    int left_ele = thread_size % concurrency_size;

    for (int i = 0; i < concurrency_size; i++) {
      int start_id = -1, end_id = -1;
      if (i < left_ele) {
        start_id = i * (avg_ele + 1);
        end_id = start_id + avg_ele + 1;
      } else {
        start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
        end_id = start_id + avg_ele;
      }

      threads_vec[i] = std::thread(
          [sequence_width,
           context_length,
           context_start,
           &lod_level_0,
           &in,
           &col,
           &im_datas,
           &col_datas,
           &col_height,
           &im_height](int start_id, int end_id) {
            for (int t = start_id; t < end_id; t++) {
              if (lod_level_0[t] == lod_level_0[t + 1]) {
                im_datas[t] = nullptr;
                col_datas[t] = nullptr;
                return;
              }
              int input_row_begin =
                  (context_start > 0)
                      ? static_cast<int>(lod_level_0[t]) + context_start
                      : static_cast<int>(lod_level_0[t]);
              int input_row_end = static_cast<int>(lod_level_0[t + 1]);

              phi::DenseTensor out_t =
                  col->Slice(static_cast<int>(lod_level_0[t]),
                             static_cast<int>(lod_level_0[t + 1]));

              col_datas[t] = out_t.data<float>();
              int sequence_height = static_cast<int>(out_t.dims()[0]);

              if (input_row_begin < input_row_end) {
                phi::DenseTensor in_t =
                    in.Slice(input_row_begin, input_row_end);
                col_height[t] = sequence_height;
                im_datas[t] = in_t.data<float>();
                im_height[t] = input_row_end - input_row_begin;
              }
            }
          },
          start_id,
          end_id);
    }
    for (int i = 0; i < concurrency_size; i++) {
      if (threads_vec[i].joinable()) threads_vec[i].join();
    }

    max_col_height = *std::max_element(col_height.begin(), col_height.end());
    // === kernel ===
    auto gpu_place = context.GetPlace();
    auto all_hbm = memory::Alloc(
        gpu_place,
        (4 * thread_size + 1 + (4 * thread_size + 1) % 2) * sizeof(uint64_t));

    int* im_height_data = reinterpret_cast<int*>(all_hbm->ptr());
    int* col_height_data = reinterpret_cast<int*>(im_height_data + thread_size);
    size_t* lod_level_0_data =
        reinterpret_cast<size_t*>(col_height_data + thread_size);
    float** im_data =
        reinterpret_cast<float**>(lod_level_0_data + thread_size + 1);
    float** col_data = reinterpret_cast<float**>(im_data + thread_size);

    // 其实im_height 就是col_height，这块可以继续优化
    cudaMemcpy(im_height_data,
               im_height.data(),
               thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_height_data,
               col_height.data(),
               thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lod_level_0_data,
               lod_level_0.data(),
               (thread_size + 1) * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(im_data,
               im_datas.data(),
               thread_size * sizeof(float*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_data,
               col_datas.data(),
               thread_size * sizeof(float*),
               cudaMemcpyHostToDevice);

    im2col_ocf_fuse(context,
                    im_data,
                    thread_size,
                    filter_height,
                    filter_width,
                    im_width,
                    col_width,
                    max_col_height,
                    im_channels,
                    col_height_data,
                    im_height_data,
                    lod_level_0_data,
                    dilation,
                    stride,
                    padding,
                    col_data);
    // === kernel ===
    if (padding_trainable) {
      PADDLE_ENFORCE_NOT_NULL(
          padding_data,
          platform::errors::InvalidArgument(
              "The input tensor 'padding_data' should not be NULL."));
      for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
        if (lod_level_0[i] == lod_level_0[i + 1]) continue;

        phi::DenseTensor out_t =
            col->Slice(static_cast<int>(lod_level_0[i]),
                       static_cast<int>(lod_level_0[i + 1]));

        int sequence_height = static_cast<int>(out_t.dims()[0]);

        // add up trainable data
        out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                      sequence_width});

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            phi::DenseTensor out_t_sub = out_t.Slice(
                k * context_length, k * context_length + padding_size);
            phi::DenseTensor w_sub = padding_data->Slice(k, k + padding_size);
            framework::TensorCopy(
                w_sub, context.GetPlace(), context, &out_t_sub);
          }
        }
        if (down_pad > 0) {  // add down pad
          int down_pad_begin_row =
              std::max(0,
                       (sequence_height - context_start - context_length) + 1) +
              1;
          int padding_begin = std::max(0, context_start - sequence_height);
          int padding_size =
              sequence_height - context_start >= context_length
                  ? 1
                  : context_length - (sequence_height - context_start);
          if (context_start >= sequence_height) padding_size = context_length;
          int padding_idx = padding_begin;
          for (int t = 0; t + down_pad_begin_row <= sequence_height;
               ++t, ++padding_size) {
            if (context_start >= sequence_height) padding_size = context_length;
            if (padding_size > context_length) {
              padding_size = context_length;
              padding_idx++;
            }
            if (padding_begin > 0 || sequence_height == context_start)
              padding_idx = padding_begin + t;

            phi::DenseTensor out_t_sub = out_t.Slice(
                (down_pad_begin_row + t) * context_length - padding_size,
                (down_pad_begin_row + t) * context_length);
            phi::DenseTensor w_sub = padding_data->Slice(
                up_pad + padding_idx, up_pad + padding_idx + padding_size);
            framework::TensorCopy(
                w_sub, context.GetPlace(), context, &out_t_sub);
          }
        }
        out_t.Resize({sequence_height,
                      static_cast<int64_t>(context_length) * sequence_width});
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ContextProjectGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& in,
                  bool padding_trainable,
                  const int context_start,
                  const int context_length,
                  const int context_stride,
                  const int up_pad,
                  const int down_pad,
                  bool pad_grad,
                  bool input_grad,
                  phi::DenseTensor* padding_data,
                  phi::DenseTensor* col) {
    auto lod_level_0 = in.lod()[0];

    phi::funcs::
        Col2ImFuseFunctor<phi::funcs::ColFormat::kOCF, DeviceContext, float>
            col2im_ocf_fuse;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    int sequence_width;
    sequence_width = in.dims()[1];
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);

    if (input_grad) {
      int concurrency_size = 5;
      int thread_size = static_cast<int>(lod_level_0.size()) - 1;
      std::vector<std::thread> threads_vec(concurrency_size);

      int im_channels = 1;
      std::vector<int> im_height(thread_size, 0);
      int im_width = sequence_width;
      int filter_height = context_length;
      int filter_width = sequence_width;
      int col_width = 1;
      std::vector<int> col_height(thread_size, 0);

      // im_data, col_data
      std::vector<float*> im_datas(thread_size);
      std::vector<float*> col_datas(thread_size);
      int max_col_height = -1;

      int avg_ele = thread_size / concurrency_size;
      int left_ele = thread_size % concurrency_size;
      for (int i = 0; i < concurrency_size; i++) {
        int start_id = -1, end_id = -1;
        if (i < left_ele) {
          start_id = i * (avg_ele + 1);
          end_id = start_id + avg_ele + 1;
        } else {
          start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
          end_id = start_id + avg_ele;
        }

        threads_vec[i] = std::thread(
            [sequence_width,
             context_length,
             context_start,
             &lod_level_0,
             &in,
             &col,
             &im_datas,
             &col_datas,
             &col_height,
             &im_height](int start_id, int end_id) {
              for (int t = start_id; t < end_id; t++) {
                if (lod_level_0[t] == lod_level_0[t + 1]) {
                  im_datas[t] = nullptr;
                  col_datas[t] = nullptr;
                  return;
                }
                int input_row_begin =
                    (context_start > 0)
                        ? static_cast<int>(lod_level_0[t]) + context_start
                        : static_cast<int>(lod_level_0[t]);
                int input_row_end = static_cast<int>(lod_level_0[t + 1]);

                phi::DenseTensor out_t =
                    col->Slice(static_cast<int>(lod_level_0[t]),
                               static_cast<int>(lod_level_0[t + 1]));

                col_datas[t] = out_t.data<float>();
                int sequence_height = static_cast<int>(out_t.dims()[0]);

                if (input_row_begin < input_row_end) {
                  phi::DenseTensor in_t =
                      in.Slice(input_row_begin, input_row_end);

                  col_height[t] = sequence_height;
                  im_datas[t] = in_t.data<float>();
                  im_height[t] = input_row_end - input_row_begin;
                }
              }
            },
            start_id,
            end_id);
      }

      for (int i = 0; i < concurrency_size; i++) {
        if (threads_vec[i].joinable()) threads_vec[i].join();
      }

      threads_vec.clear();
      threads_vec.resize(concurrency_size);
      max_col_height = *std::max_element(col_height.begin(), col_height.end());

      auto gpu_place = context.GetPlace();
      auto all_hbm = memory::Alloc(
          gpu_place,
          (4 * thread_size + 1 + (4 * thread_size + 1) % 2) * sizeof(uint64_t));

      int* im_height_data = reinterpret_cast<int*>(all_hbm->ptr());
      int* col_height_data =
          reinterpret_cast<int*>(im_height_data + thread_size);
      size_t* lod_level_0_data =
          reinterpret_cast<size_t*>(col_height_data + thread_size);
      float** im_data =
          reinterpret_cast<float**>(lod_level_0_data + thread_size + 1);
      float** col_data = reinterpret_cast<float**>(im_data + thread_size);

      // 其实im_height 就是col_height，这块可以继续优化
      cudaMemcpy(im_height_data,
                 im_height.data(),
                 thread_size * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(col_height_data,
                 col_height.data(),
                 thread_size * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(lod_level_0_data,
                 lod_level_0.data(),
                 (thread_size + 1) * sizeof(size_t),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(im_data,
                 im_datas.data(),
                 thread_size * sizeof(float*),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(col_data,
                 col_datas.data(),
                 thread_size * sizeof(float*),
                 cudaMemcpyHostToDevice);
      col2im_ocf_fuse(context,
                      col_data,
                      thread_size,
                      filter_height,
                      filter_width,
                      im_width,
                      col_width,
                      max_col_height,
                      im_channels,
                      col_height_data,
                      im_height_data,
                      lod_level_0_data,
                      dilation,
                      stride,
                      padding,
                      im_data);
    }
    if (pad_grad) {
      if (padding_trainable) {
        for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
          if (lod_level_0[i] == lod_level_0[i + 1]) continue;

          phi::DenseTensor out_t =
              col->Slice(static_cast<int>(lod_level_0[i]),
                         static_cast<int>(lod_level_0[i + 1]));

          int sequence_height = static_cast<int>(out_t.dims()[0]);
          out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                        sequence_width});

          if (up_pad > 0) {
            int padding_rows = std::min(
                up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

            for (int k = 0; k < padding_rows; ++k) {
              int padding_size =
                  k + context_length < up_pad ? context_length : up_pad - k;
              phi::DenseTensor out_t_sub = out_t.Slice(
                  k * context_length, k * context_length + padding_size);
              phi::DenseTensor w_sub = padding_data->Slice(k, k + padding_size);
              blas.AXPY(w_sub.numel(),
                        static_cast<T>(1),
                        out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          if (down_pad > 0) {
            int down_pad_begin_row =
                std::max(
                    0, (sequence_height - context_start - context_length) + 1) +
                1;
            int padding_begin = std::max(0, context_start - sequence_height);
            int padding_size =
                sequence_height - context_start >= context_length
                    ? 1
                    : context_length - (sequence_height - context_start);
            if (context_start >= sequence_height) padding_size = context_length;
            int padding_idx = padding_begin;
            for (int t = 0; t + down_pad_begin_row <= sequence_height;
                 ++t, ++padding_size) {
              if (context_start >= sequence_height)
                padding_size = context_length;
              if (padding_size > context_length) {
                padding_size = context_length;
                padding_idx++;
              }
              if (padding_begin > 0 || sequence_height == context_start)
                padding_idx = padding_begin + t;

              phi::DenseTensor out_t_sub = out_t.Slice(
                  (down_pad_begin_row + t) * context_length - padding_size,
                  (down_pad_begin_row + t) * context_length);
              phi::DenseTensor w_sub = padding_data->Slice(
                  up_pad + padding_idx, up_pad + padding_idx + padding_size);
              blas.AXPY(w_sub.numel(),
                        static_cast<T>(1),
                        out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          out_t.Resize({sequence_height,
                        static_cast<int64_t>(context_length) * sequence_width});
        }
      }
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
