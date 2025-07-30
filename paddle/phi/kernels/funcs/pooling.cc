/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/pooling.h"
#include <algorithm>
#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi::funcs {

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dFunctor<CPUContext, PoolProcess, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NHWC");

    const int64_t batch_size = input.dims()[0];
    const int64_t input_channels =
        channel_last ? input.dims()[3] : input.dims()[1];
    const int64_t input_height =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_width =
        channel_last ? input.dims()[2] : input.dims()[3];

    const int64_t output_channels =
        channel_last ? output->dims()[3] : output->dims()[1];
    const int64_t output_height =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int64_t output_width =
        channel_last ? output->dims()[2] : output->dims()[3];

    const int64_t ksize_height = ksize[0];
    const int64_t ksize_width = ksize[1];

    const int64_t stride_height = strides[0];
    const int64_t stride_width = strides[1];

    const int64_t padding_height = paddings[0];
    const int64_t padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int64_t hstart = 0, hend = 1;
    int64_t wstart = 0, wend = 1;
    if (!channel_last) {
      const int64_t input_stride = input_height * input_width;
      const int64_t output_stride = output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, static_cast<int64_t>(0));
                hstart = std::max(hstart, static_cast<int64_t>(0));
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }

              T ele = pool_process.initial();
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  pool_process.compute(input_data[h * input_width + w], &ele);
                }
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              pool_process.finalize(static_cast<T>(pool_size), &ele);
              output_data[ph * output_width + pw] = ele;
            }
          }
          input_data += input_stride;
          output_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride = input_height * input_width * input_channels;
      const int64_t output_stride =
          output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, static_cast<int64_t>(0));
                hstart = std::max(hstart, static_cast<int64_t>(0));
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              T ele = pool_process.initial();
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  pool_process.compute(
                      input_data[h * input_width * input_channels +
                                 w * input_channels + c],
                      &ele);
                }
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              pool_process.finalize(static_cast<T>(pool_size), &ele);
              output_data[ph * output_width * output_channels +
                          pw * output_channels + c] = ele;
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

/*
 * tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool2dGradFunctor<CPUContext, PoolProcess, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_grad_process) {
    bool channel_last = (data_format == "NHWC");

    const int64_t batch_size = input.dims()[0];

    const int64_t input_channels =
        channel_last ? input.dims()[3] : input.dims()[1];
    const int64_t input_height =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_width =
        channel_last ? input.dims()[2] : input.dims()[3];

    const int64_t output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int64_t output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int64_t output_width =
        channel_last ? output.dims()[2] : output.dims()[3];

    const int64_t ksize_height = ksize[0];
    const int64_t ksize_width = ksize[1];

    const int64_t stride_height = strides[0];
    const int64_t stride_width = strides[1];

    const int64_t padding_height = paddings[0];
    const int64_t padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int64_t hstart = 0, hend = 1;
    int64_t wstart = 0, wend = 1;
    if (!channel_last) {
      const int64_t input_stride = input_height * input_width;
      const int64_t output_stride = output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, static_cast<int64_t>(0));
                hstart = std::max(hstart, static_cast<int64_t>(0));
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              float scale = 1.0f / static_cast<float>(pool_size);
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  pool_grad_process.compute(
                      input_data[h * input_width + w],
                      output_data[ph * output_width + pw],
                      output_grad_data[ph * output_width + pw],
                      static_cast<T>(scale),
                      input_grad_data + h * input_width + w);
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride = input_height * input_width * input_channels;
      const int64_t output_stride =
          output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, static_cast<int64_t>(0));
                hstart = std::max(hstart, static_cast<int64_t>(0));
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              float scale = 1.0f / static_cast<float>(pool_size);
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  auto input_idx =
                      h * input_width * input_channels + w * input_channels + c;
                  auto output_idx = ph * output_width * output_channels +
                                    pw * output_channels + c;
                  pool_grad_process.compute(input_data[input_idx],
                                            output_data[output_idx],
                                            output_grad_data[output_idx],
                                            static_cast<T>(scale),
                                            input_grad_data + input_idx);
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <class T>
class MaxPool2dGradFunctor<CPUContext, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad) {
    bool channel_last = (data_format == "NHWC");

    const int64_t batch_size = input.dims()[0];

    const int64_t input_channels =
        channel_last ? input.dims()[3] : input.dims()[1];
    const int64_t input_height =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_width =
        channel_last ? input.dims()[2] : input.dims()[3];

    const int64_t output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int64_t output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int64_t output_width =
        channel_last ? output.dims()[2] : output.dims()[3];

    const int64_t ksize_height = ksize[0];
    const int64_t ksize_width = ksize[1];

    const int64_t stride_height = strides[0];
    const int64_t stride_width = strides[1];

    const int64_t padding_height = paddings[0];
    const int64_t padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    if (!channel_last) {
      const int64_t input_stride = input_height * input_width;
      const int64_t output_stride = output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            int64_t hstart = ph * stride_height - padding_height;
            int64_t hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t wstart = pw * stride_width - padding_width;
              int64_t wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, static_cast<int64_t>(0));

              bool stop = false;
              for (int64_t h = hstart; h < hend && !stop; ++h) {
                for (int64_t w = wstart; w < wend && !stop; ++w) {
                  int64_t input_idx = h * input_width + w;
                  int64_t output_idx = ph * output_width + pw;
                  if (input_data[input_idx] == output_data[output_idx]) {
                    input_grad_data[input_idx] += output_grad_data[output_idx];
                    stop = true;
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride = input_height * input_width * input_channels;
      const int64_t output_stride =
          output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            int64_t hstart = ph * stride_height - padding_height;
            int64_t hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < output_width; ++pw) {
              int64_t wstart = pw * stride_width - padding_width;
              int64_t wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, static_cast<int64_t>(0));

              bool stop = false;
              for (int64_t h = hstart; h < hend && !stop; ++h) {
                for (int64_t w = wstart; w < wend && !stop; ++w) {
                  int64_t input_idx =
                      h * input_width * input_channels + w * input_channels + c;
                  int64_t output_idx = ph * output_width * output_channels +
                                       pw * output_channels + c;
                  if (input_data[input_idx] == output_data[output_idx]) {
                    input_grad_data[input_idx] += output_grad_data[output_idx];
                    stop = true;
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};
template class MaxPool2dGradFunctor<CPUContext, float>;
template class MaxPool2dGradFunctor<CPUContext, double>;

template class Pool2dFunctor<CPUContext, MaxPool<float>, float>;
template class Pool2dFunctor<CPUContext, AvgPool<float>, float>;
template class Pool2dFunctor<CPUContext, LPPool<float>, float>;
template class Pool2dGradFunctor<CPUContext, MaxPoolGrad<float>, float>;
template class Pool2dGradFunctor<CPUContext, AvgPoolGrad<float>, float>;
template class Pool2dGradFunctor<CPUContext, LPPoolGrad<float>, float>;
template class Pool2dFunctor<CPUContext, MaxPool<double>, double>;
template class Pool2dFunctor<CPUContext, AvgPool<double>, double>;
template class Pool2dFunctor<CPUContext, LPPool<double>, double>;
template class Pool2dGradFunctor<CPUContext, MaxPoolGrad<double>, double>;
template class Pool2dGradFunctor<CPUContext, AvgPoolGrad<double>, double>;
template class Pool2dGradFunctor<CPUContext, LPPoolGrad<double>, double>;

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dFunctor<CPUContext, PoolProcess, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NDHWC");
    const int64_t batch_size = input.dims()[0];

    const int64_t input_channels =
        channel_last ? input.dims()[4] : input.dims()[1];
    const int64_t input_depth =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_height =
        channel_last ? input.dims()[2] : input.dims()[3];
    const int64_t input_width =
        channel_last ? input.dims()[3] : input.dims()[4];

    const int64_t output_channels =
        channel_last ? output->dims()[4] : output->dims()[1];
    const int64_t output_depth =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int64_t output_height =
        channel_last ? output->dims()[2] : output->dims()[3];
    const int64_t output_width =
        channel_last ? output->dims()[3] : output->dims()[4];

    const int64_t ksize_depth = ksize[0];
    const int64_t ksize_height = ksize[1];
    const int64_t ksize_width = ksize[2];

    const int64_t stride_depth = strides[0];
    const int64_t stride_height = strides[1];
    const int64_t stride_width = strides[2];

    const int64_t padding_depth = paddings[0];
    const int64_t padding_height = paddings[1];
    const int64_t padding_width = paddings[2];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int64_t dstart = 0, dend = 1;
    int64_t hstart = 0, hend = 1;
    int64_t wstart = 0, wend = 1;
    if (!channel_last) {
      const int64_t input_stride = input_depth * input_height * input_width;
      const int64_t output_stride = output_depth * output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int64_t ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, static_cast<int64_t>(0));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                int64_t output_idx =
                    (pd * output_height + ph) * output_width + pw;
                T ele = pool_process.initial();
                for (int64_t d = dstart; d < dend; ++d) {
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      pool_process.compute(
                          input_data[(d * input_height + h) * input_width + w],
                          &ele);
                    }
                  }
                }
                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                pool_process.finalize(static_cast<T>(pool_size), &ele);
                output_data[output_idx] = ele;
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride =
          input_depth * input_height * input_width * input_channels;
      const int64_t output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int64_t ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, static_cast<int64_t>(0));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                T ele = pool_process.initial();
                for (int64_t d = dstart; d < dend; ++d) {
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      int64_t input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      pool_process.compute(input_data[input_idx], &ele);
                    }
                  }
                }
                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                pool_process.finalize(static_cast<T>(pool_size), &ele);
                int64_t output_idx =
                    ((pd * output_height + ph) * output_width + pw) *
                        output_channels +
                    c;
                output_data[output_idx] = ele;
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dGradFunctor<CPUContext, PoolProcess, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_grad_process) {
    bool channel_last = (data_format == "NDHWC");

    const int64_t batch_size = input.dims()[0];
    const int64_t input_channels =
        channel_last ? input.dims()[4] : input.dims()[1];
    const int64_t input_depth =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_height =
        channel_last ? input.dims()[2] : input.dims()[3];
    const int64_t input_width =
        channel_last ? input.dims()[3] : input.dims()[4];

    const int64_t output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int64_t output_depth =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int64_t output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int64_t output_width =
        channel_last ? output.dims()[3] : output.dims()[4];

    const int64_t ksize_depth = ksize[0];
    const int64_t ksize_height = ksize[1];
    const int64_t ksize_width = ksize[2];

    const int64_t stride_depth = strides[0];
    const int64_t stride_height = strides[1];
    const int64_t stride_width = strides[2];

    const int64_t padding_depth = paddings[0];
    const int64_t padding_height = paddings[1];
    const int64_t padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int64_t dstart = 0, dend = 1;
    int64_t hstart = 0, hend = 1;
    int64_t wstart = 0, wend = 1;
    if (!channel_last) {
      const int64_t input_stride = input_depth * input_height * input_width;
      const int64_t output_stride = output_depth * output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int64_t ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, static_cast<int64_t>(0));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                float scale = 1.0f / static_cast<float>(pool_size);
                for (int64_t d = dstart; d < dend; ++d) {
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      int64_t input_idx =
                          (d * input_height + h) * input_width + w;
                      int64_t output_idx =
                          (pd * output_height + ph) * output_width + pw;
                      pool_grad_process.compute(input_data[input_idx],
                                                output_data[output_idx],
                                                output_grad_data[output_idx],
                                                static_cast<T>(scale),
                                                input_grad_data + input_idx);
                    }
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride =
          input_depth * input_height * input_width * input_channels;
      const int64_t output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int64_t ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, static_cast<int64_t>(0));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                float scale = 1.0f / static_cast<float>(pool_size);
                for (int64_t d = dstart; d < dend; ++d) {
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      int64_t input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      int64_t output_idx =
                          ((pd * output_height + ph) * output_width + pw) *
                              output_channels +
                          c;
                      pool_grad_process.compute(input_data[input_idx],
                                                output_data[output_idx],
                                                output_grad_data[output_idx],
                                                static_cast<T>(scale),
                                                input_grad_data + input_idx);
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <class T>
class MaxPool3dGradFunctor<CPUContext, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad) {
    bool channel_last = (data_format == "NDHWC");
    const int64_t batch_size = input.dims()[0];

    const int64_t input_channels =
        channel_last ? input.dims()[4] : input.dims()[1];
    const int64_t input_depth =
        channel_last ? input.dims()[1] : input.dims()[2];
    const int64_t input_height =
        channel_last ? input.dims()[2] : input.dims()[3];
    const int64_t input_width =
        channel_last ? input.dims()[3] : input.dims()[4];

    const int64_t output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int64_t output_depth =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int64_t output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int64_t output_width =
        channel_last ? output.dims()[3] : output.dims()[4];

    const int64_t ksize_depth = ksize[0];
    const int64_t ksize_height = ksize[1];
    const int64_t ksize_width = ksize[2];

    const int64_t stride_depth = strides[0];
    const int64_t stride_height = strides[1];
    const int64_t stride_width = strides[2];

    const int64_t padding_depth = paddings[0];
    const int64_t padding_height = paddings[1];
    const int64_t padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    if (!channel_last) {
      const int64_t input_stride = input_depth * input_height * input_width;
      const int64_t output_stride = output_depth * output_height * output_width;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            int64_t dstart = pd * stride_depth - padding_depth;
            int64_t dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, static_cast<int64_t>(0));
            for (int64_t ph = 0; ph < output_height; ++ph) {
              int64_t hstart = ph * stride_height - padding_height;
              int64_t hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, static_cast<int64_t>(0));
              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t wstart = pw * stride_width - padding_width;
                int64_t wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, static_cast<int64_t>(0));
                bool stop = false;
                for (int64_t d = dstart; d < dend && !stop; ++d) {
                  for (int64_t h = hstart; h < hend && !stop; ++h) {
                    for (int64_t w = wstart; w < wend && !stop; ++w) {
                      int64_t input_idx =
                          (d * input_height + h) * input_width + w;
                      int64_t output_idx =
                          (pd * output_height + ph) * output_width + pw;

                      if (input_data[input_idx] == output_data[output_idx]) {
                        input_grad_data[input_idx] +=
                            output_grad_data[output_idx];
                        stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int64_t input_stride =
          input_depth * input_height * input_width * input_channels;
      const int64_t output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t c = 0; c < output_channels; ++c) {
          for (int64_t pd = 0; pd < output_depth; ++pd) {
            int64_t dstart = pd * stride_depth - padding_depth;
            int64_t dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, static_cast<int64_t>(0));
            for (int64_t ph = 0; ph < output_height; ++ph) {
              int64_t hstart = ph * stride_height - padding_height;
              int64_t hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, static_cast<int64_t>(0));
              for (int64_t pw = 0; pw < output_width; ++pw) {
                int64_t wstart = pw * stride_width - padding_width;
                int64_t wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, static_cast<int64_t>(0));
                bool stop = false;

                for (int64_t d = dstart; d < dend && !stop; ++d) {
                  for (int64_t h = hstart; h < hend && !stop; ++h) {
                    for (int64_t w = wstart; w < wend && !stop; ++w) {
                      int64_t input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      int64_t output_idx =
                          ((pd * output_height + ph) * output_width + pw) *
                              output_channels +
                          c;

                      if (input_data[input_idx] == output_data[output_idx]) {
                        input_grad_data[input_idx] +=
                            output_grad_data[output_idx];
                        stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};
template class MaxPool3dGradFunctor<CPUContext, float>;
template class MaxPool3dGradFunctor<CPUContext, double>;

template class Pool3dFunctor<CPUContext, MaxPool<float>, float>;
template class Pool3dFunctor<CPUContext, AvgPool<float>, float>;
template class Pool3dGradFunctor<CPUContext, MaxPoolGrad<float>, float>;
template class Pool3dGradFunctor<CPUContext, AvgPoolGrad<float>, float>;
template class Pool3dFunctor<CPUContext, MaxPool<double>, double>;
template class Pool3dFunctor<CPUContext, AvgPool<double>, double>;
template class Pool3dGradFunctor<CPUContext, MaxPoolGrad<double>, double>;
template class Pool3dGradFunctor<CPUContext, AvgPoolGrad<double>, double>;

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int64_t batch_size = input.dims()[0];
    const int64_t input_height = input.dims()[2];
    const int64_t input_width = input.dims()[3];
    const int64_t output_channels = output->dims()[1];
    const int64_t output_height = output->dims()[2];
    const int64_t output_width = output->dims()[3];
    const int64_t ksize_height = ksize[0];
    const int64_t ksize_width = ksize[1];
    const int64_t stride_height = strides[0];
    const int64_t stride_width = strides[1];
    const int64_t padding_height = paddings[0];
    const int64_t padding_width = paddings[1];
    const int64_t input_stride = input_height * input_width;
    const int64_t output_stride = output_height * output_width;

    const T1* input_data = input.data<T1>();
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    int64_t hstart = 0, hend = 0;
    int64_t wstart = 0, wend = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t ph = 0; ph < output_height; ++ph) {
          if (adaptive) {
            hstart = AdaptStartIndex(ph, input_height, output_height);
            hend = AdaptEndIndex(ph, input_height, output_height);
          } else {
            hstart = ph * stride_height - padding_height;
            hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
          }
          for (int64_t pw = 0; pw < output_width; ++pw) {
            if (adaptive) {
              wstart = AdaptStartIndex(pw, input_width, output_width);
              wend = AdaptEndIndex(pw, input_width, output_width);
            } else {
              wstart = pw * stride_width - padding_width;
              wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
            }

            T1 ele = static_cast<T1>(-FLT_MAX);
            int64_t index = -1;
            for (int64_t h = hstart; h < hend; ++h) {
              for (int64_t w = wstart; w < wend; ++w) {
                if (ele < input_data[h * input_width + w]) {
                  ele = input_data[h * input_width + w];
                  index = h * input_width + w;
                }
              }
            }
            output_data[ph * output_width + pw] = ele;
            mask_data[ph * output_width + pw] = index;
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int64_t>& ksize UNUSED,
                  const std::vector<int64_t>& strides UNUSED,
                  const std::vector<int64_t>& paddings UNUSED,
                  bool adaptive UNUSED,
                  DenseTensor* input_grad) {
    const int64_t batch_size = input_grad->dims()[0];
    const int64_t input_height = input_grad->dims()[2];
    const int64_t input_width = input_grad->dims()[3];
    const int64_t output_channels = output_grad.dims()[1];
    const int64_t output_height = output_grad.dims()[2];
    const int64_t output_width = output_grad.dims()[3];
    const int64_t input_stride = input_height * input_width;
    const int64_t output_stride = output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t ph = 0; ph < output_height; ++ph) {
          for (int64_t pw = 0; pw < output_width; ++pw) {
            const int64_t output_idx = ph * output_width + pw;
            const int64_t input_idx =
                static_cast<int64_t>(mask_data[output_idx]);
            input_grad_data[input_idx] += output_grad_data[output_idx];
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool2dWithIndexFunctor<CPUContext, float, int>;
template class MaxPool2dWithIndexGradFunctor<CPUContext, float, int>;
template class MaxPool2dWithIndexFunctor<CPUContext, double, int>;
template class MaxPool2dWithIndexGradFunctor<CPUContext, double, int>;

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& ksize,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int64_t batch_size = input.dims()[0];
    const int64_t input_depth = input.dims()[2];
    const int64_t input_height = input.dims()[3];
    const int64_t input_width = input.dims()[4];
    const int64_t output_channels = output->dims()[1];
    const int64_t output_depth = output->dims()[2];
    const int64_t output_height = output->dims()[3];
    const int64_t output_width = output->dims()[4];
    const int64_t ksize_depth = ksize[0];
    const int64_t ksize_height = ksize[1];
    const int64_t ksize_width = ksize[2];
    const int64_t stride_depth = strides[0];
    const int64_t stride_height = strides[1];
    const int64_t stride_width = strides[2];
    const int64_t padding_depth = paddings[0];
    const int64_t padding_height = paddings[1];
    const int64_t padding_width = paddings[2];
    const int64_t input_stride = input_depth * input_height * input_width;
    const int64_t output_stride = output_depth * output_height * output_width;

    const T1* input_data = input.data<T1>();
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    int64_t dstart = 0, dend = 0;
    int64_t hstart = 0, hend = 0;
    int64_t wstart = 0, wend = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t pd = 0; pd < output_depth; ++pd) {
          if (adaptive) {
            dstart = AdaptStartIndex(pd, input_depth, output_depth);
            dend = AdaptEndIndex(pd, input_depth, output_depth);
          } else {
            dstart = pd * stride_depth - padding_depth;
            dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, static_cast<int64_t>(0));
          }
          for (int64_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            } else {
              hstart = ph * stride_height - padding_height;
              hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, static_cast<int64_t>(0));
            }
            for (int64_t pw = 0; pw < output_width; ++pw) {
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                wstart = pw * stride_width - padding_width;
                wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, static_cast<int64_t>(0));
              }

              int64_t output_idx =
                  (pd * output_height + ph) * output_width + pw;
              T1 ele = static_cast<T1>(-FLT_MAX);
              int64_t index = -1;
              for (int64_t d = dstart; d < dend; ++d) {
                for (int64_t h = hstart; h < hend; ++h) {
                  for (int64_t w = wstart; w < wend; ++w) {
                    int64_t input_idx =
                        (d * input_height + h) * input_width + w;
                    if (ele < input_data[input_idx]) {
                      index = input_idx;
                      ele = input_data[input_idx];
                    }
                  }
                }
              }
              output_data[output_idx] = ele;
              mask_data[output_idx] = index;
            }
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int64_t>& ksize UNUSED,
                  const std::vector<int64_t>& strides UNUSED,
                  const std::vector<int64_t>& paddings UNUSED,
                  bool adaptive UNUSED,
                  DenseTensor* input_grad) {
    const int64_t batch_size = input_grad->dims()[0];
    const int64_t input_depth = input_grad->dims()[2];
    const int64_t input_height = input_grad->dims()[3];
    const int64_t input_width = input_grad->dims()[4];
    const int64_t output_channels = output_grad.dims()[1];
    const int64_t output_depth = output_grad.dims()[2];
    const int64_t output_height = output_grad.dims()[3];
    const int64_t output_width = output_grad.dims()[4];
    const int64_t input_stride = input_depth * input_height * input_width;
    const int64_t output_stride = output_depth * output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t pd = 0; pd < output_depth; ++pd) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            for (int64_t pw = 0; pw < output_width; ++pw) {
              const int64_t output_idx =
                  (pd * output_height + ph) * output_width + pw;
              const int64_t input_idx =
                  static_cast<int64_t>(mask_data[output_idx]);
              input_grad_data[input_idx] += output_grad_data[output_idx];
            }
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool3dWithIndexFunctor<CPUContext, float, int>;
template class MaxPool3dWithIndexGradFunctor<CPUContext, float, int>;
template class MaxPool3dWithIndexFunctor<CPUContext, double, int>;
template class MaxPool3dWithIndexGradFunctor<CPUContext, double, int>;

/*
 * All tensors are in NCHW format.
 */
template <typename T1, typename T2>
class FractionalMaxPool2dFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& output_size,
                  const std::vector<int64_t>& kernel_size,
                  float random_u,
                  bool return_mask,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int64_t batch_size = input.dims()[0];
    const int64_t input_height = input.dims()[2];
    const int64_t input_width = input.dims()[3];
    const int64_t output_channels = output->dims()[1];
    const int64_t output_height = output->dims()[2];
    const int64_t output_width = output->dims()[3];
    const int64_t pool_height = kernel_size[0];
    const int64_t pool_width = kernel_size[1];
    const int64_t input_stride = input_height * input_width;
    const int64_t output_stride = output_height * output_width;

    PADDLE_ENFORCE_GE(
        input_height,
        output_height - 1 + pool_height,
        common::errors::InvalidArgument(
            "input_height [%d] is less than valid output_height [%d]",
            input_height,
            output_height - 1 + pool_height));
    PADDLE_ENFORCE_GE(
        input_width,
        output_width - 1 + pool_width,
        common::errors::InvalidArgument(
            "input_width [%d] is less than valid output_width [%d]",
            input_width,
            output_width - 1 + pool_width));

    const T1* input_data = input.data<T1>();
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    float alpha_height = 0, alpha_width = 0;
    float u_height = 0, u_width = 0;
    float u = 0;
    if (random_u == 0) {
      std::uniform_real_distribution<float> dist(0, 1);
      auto engine = phi::GetCPURandomEngine(0);
      u = dist(*engine);
    } else {
      u = random_u;
    }

    alpha_height = static_cast<float>(input_height - pool_height) /
                   (output_height - (pool_height > 0 ? 1 : 0));
    alpha_width = static_cast<float>(input_width - pool_width) /
                  (output_width - (pool_width > 0 ? 1 : 0));

    u_height = FractionalRationalU(
        u, alpha_height, input_height, output_height, pool_height);
    u_width = FractionalRationalU(
        u, alpha_width, input_width, output_width, pool_width);

    int64_t hstart = 0, hend = 0;
    int64_t wstart = 0, wend = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t ph = 0; ph < output_height; ++ph) {
          hstart =
              FractionalStartIndex(ph, alpha_height, u_height, pool_height);
          hend = FractionalEndIndex(ph, alpha_height, u_height, pool_height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          hend = std::min(hend, input_height);

          for (int64_t pw = 0; pw < output_width; ++pw) {
            wstart = FractionalStartIndex(pw, alpha_width, u_width, pool_width);
            wend = FractionalEndIndex(pw, alpha_width, u_width, pool_width);
            wstart = std::max(wstart, static_cast<int64_t>(0));
            wend = std::min(wend, input_width);

            T1 ele = static_cast<T1>(-FLT_MAX);
            int64_t index = -1;
            for (int64_t h = hstart; h < hend; ++h) {
              for (int64_t w = wstart; w < wend; ++w) {
                if (ele < input_data[h * input_width + w]) {
                  ele = input_data[h * input_width + w];
                  index = h * input_width + w;
                }
              }
            }
            output_data[ph * output_width + pw] = ele;
            mask_data[ph * output_width + pw] = index;
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCHW format.
 */
template <typename T1, typename T2>
class FractionalMaxPool2dGradFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int64_t>& output_size UNUSED,
                  const std::vector<int64_t>& kernel_size UNUSED,
                  float random_u UNUSED,
                  bool return_mask UNUSED,
                  DenseTensor* input_grad) {
    const int64_t batch_size = input_grad->dims()[0];
    const int64_t input_height = input_grad->dims()[2];
    const int64_t input_width = input_grad->dims()[3];
    const int64_t output_channels = output_grad.dims()[1];
    const int64_t output_height = output_grad.dims()[2];
    const int64_t output_width = output_grad.dims()[3];
    const int64_t input_stride = input_height * input_width;
    const int64_t output_stride = output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t ph = 0; ph < output_height; ++ph) {
          for (int64_t pw = 0; pw < output_width; ++pw) {
            const int64_t output_idx = ph * output_width + pw;
            const int64_t input_idx =
                static_cast<int64_t>(mask_data[output_idx]);
            input_grad_data[input_idx] += output_grad_data[output_idx];
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class FractionalMaxPool2dFunctor<CPUContext, float, int>;
template class FractionalMaxPool2dGradFunctor<CPUContext, float, int>;
template class FractionalMaxPool2dFunctor<CPUContext, double, int>;
template class FractionalMaxPool2dGradFunctor<CPUContext, double, int>;
template class FractionalMaxPool2dFunctor<CPUContext, dtype::float16, int>;
template class FractionalMaxPool2dGradFunctor<CPUContext, dtype::float16, int>;

/*
 * All tensors are in NCDHW format.
 */
template <typename T1, typename T2>
class FractionalMaxPool3dFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int64_t>& output_size,
                  const std::vector<int64_t>& kernel_size,
                  float random_u,
                  bool return_mask,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int64_t batch_size = input.dims()[0];
    const int64_t input_depth = input.dims()[2];
    const int64_t input_height = input.dims()[3];
    const int64_t input_width = input.dims()[4];
    const int64_t output_channels = output->dims()[1];
    const int64_t output_depth = output->dims()[2];
    const int64_t output_height = output->dims()[3];
    const int64_t output_width = output->dims()[4];
    const int64_t pool_depth = kernel_size[0];
    const int64_t pool_height = kernel_size[1];
    const int64_t pool_width = kernel_size[2];
    const int64_t input_stride = input_depth * input_height * input_width;
    const int64_t output_stride = output_depth * output_height * output_width;

    PADDLE_ENFORCE_GE(
        input_depth,
        output_depth - 1 + pool_depth,
        common::errors::InvalidArgument(
            "input_depth [%d] is less than valid output_depth [%d]",
            input_depth,
            output_depth - 1 + pool_depth));
    PADDLE_ENFORCE_GE(
        input_height,
        output_height - 1 + pool_height,
        common::errors::InvalidArgument(
            "input_height [%d] is less than valid output_height [%d]",
            input_height,
            output_height - 1 + pool_height));
    PADDLE_ENFORCE_GE(
        input_width,
        output_width - 1 + pool_width,
        common::errors::InvalidArgument(
            "input_width [%d] is less than valid output_width [%d]",
            input_width,
            output_width - 1 + pool_width));

    const T1* input_data = input.data<T1>();
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    float alpha_height = 0, alpha_width = 0, alpha_depth = 0;
    float u_height = 0, u_width = 0, u_depth = 0;
    float u = 0;
    if (random_u == 0) {
      std::uniform_real_distribution<float> dist(0, 1);
      auto engine = phi::GetCPURandomEngine(0);
      u = dist(*engine);
    } else {
      u = random_u;
    }

    alpha_depth = static_cast<float>(input_depth - pool_depth) /
                  (output_depth - (pool_depth > 0 ? 1 : 0));
    alpha_height = static_cast<float>(input_height - pool_height) /
                   (output_height - (pool_height > 0 ? 1 : 0));
    alpha_width = static_cast<float>(input_width - pool_width) /
                  (output_width - (pool_width > 0 ? 1 : 0));

    u_depth = FractionalRationalU(
        u, alpha_depth, input_depth, output_depth, pool_depth);
    u_height = FractionalRationalU(
        u, alpha_height, input_height, output_height, pool_height);
    u_width = FractionalRationalU(
        u, alpha_width, input_width, output_width, pool_width);

    int64_t dstart = 0, dend = 0;
    int64_t hstart = 0, hend = 0;
    int64_t wstart = 0, wend = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t pd = 0; pd < output_depth; ++pd) {
          dstart = FractionalStartIndex(pd, alpha_depth, u_depth, pool_depth);
          dend = FractionalEndIndex(pd, alpha_depth, u_depth, pool_depth);
          dstart = std::max(dstart, static_cast<int64_t>(0));
          dend = std::min(dend, input_depth);

          for (int64_t ph = 0; ph < output_height; ++ph) {
            hstart =
                FractionalStartIndex(ph, alpha_height, u_height, pool_height);
            hend = FractionalEndIndex(ph, alpha_height, u_height, pool_height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            hend = std::min(hend, input_height);

            for (int64_t pw = 0; pw < output_width; ++pw) {
              wstart =
                  FractionalStartIndex(pw, alpha_width, u_width, pool_width);
              wend = FractionalEndIndex(pw, alpha_width, u_width, pool_width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
              wend = std::min(wend, input_width);

              int64_t output_idx =
                  (pd * output_height + ph) * output_width + pw;
              T1 ele = static_cast<T1>(-FLT_MAX);
              int64_t index = -1;
              for (int64_t d = dstart; d < dend; ++d) {
                for (int64_t h = hstart; h < hend; ++h) {
                  for (int64_t w = wstart; w < wend; ++w) {
                    int64_t input_idx =
                        (d * input_height + h) * input_width + w;
                    if (ele < input_data[input_idx]) {
                      index = input_idx;
                      ele = input_data[input_idx];
                    }
                  }
                }
              }
              output_data[output_idx] = ele;
              mask_data[output_idx] = index;
            }
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCDHW format.
 */
template <typename T1, typename T2>
class FractionalMaxPool3dGradFunctor<CPUContext, T1, T2> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int64_t>& output_size UNUSED,
                  const std::vector<int64_t>& kernel_size UNUSED,
                  float random_u UNUSED,
                  bool return_mask UNUSED,
                  DenseTensor* input_grad) {
    const int64_t batch_size = input_grad->dims()[0];
    const int64_t input_depth = input_grad->dims()[2];
    const int64_t input_height = input_grad->dims()[3];
    const int64_t input_width = input_grad->dims()[4];
    const int64_t output_channels = output_grad.dims()[1];
    const int64_t output_depth = output_grad.dims()[2];
    const int64_t output_height = output_grad.dims()[3];
    const int64_t output_width = output_grad.dims()[4];
    const int64_t input_stride = input_depth * input_height * input_width;
    const int64_t output_stride = output_depth * output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t c = 0; c < output_channels; ++c) {
        for (int64_t pd = 0; pd < output_depth; ++pd) {
          for (int64_t ph = 0; ph < output_height; ++ph) {
            for (int64_t pw = 0; pw < output_width; ++pw) {
              const int64_t output_idx =
                  (pd * output_height + ph) * output_width + pw;
              const int64_t input_idx =
                  static_cast<int64_t>(mask_data[output_idx]);
              input_grad_data[input_idx] += output_grad_data[output_idx];
            }
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class FractionalMaxPool3dFunctor<CPUContext, float, int>;
template class FractionalMaxPool3dGradFunctor<CPUContext, float, int>;
template class FractionalMaxPool3dFunctor<CPUContext, double, int>;
template class FractionalMaxPool3dGradFunctor<CPUContext, double, int>;
template class FractionalMaxPool3dFunctor<CPUContext, dtype::float16, int>;
template class FractionalMaxPool3dGradFunctor<CPUContext, dtype::float16, int>;

}  // namespace phi::funcs
