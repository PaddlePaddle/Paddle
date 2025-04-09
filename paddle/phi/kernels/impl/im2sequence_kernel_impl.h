// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/optional.h"
namespace phi {

inline int Im2SeqOutputSize(
    int input_size, int filter_size, int padding_0, int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

template <typename T, typename Context>
void Im2SequenceKernel(const Context& dev_ctx,
                       const DenseTensor& x_in,
                       const paddle::optional<DenseTensor>& y,
                       const std::vector<int>& kernels,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& out_stride,
                       DenseTensor* out) {
  const phi::DenseTensor* in = &x_in;
  auto in_dim = in->dims();
  int batch_size = in_dim[0];
  int img_channels = in_dim[1];
  int img_height = in_dim[2];
  int img_width = in_dim[3];
  if (y && batch_size > 1) {
    const phi::DenseTensor* img_real_size = y.get_ptr();

    phi::DenseTensor cpu_shape_tensor;
    phi::Copy(
        dev_ctx, *img_real_size, phi::CPUPlace(), true, &cpu_shape_tensor);
    std::vector<int> img_real_h;
    std::vector<int> img_real_w;
    std::vector<int> output_height;
    std::vector<int> output_width;
    int result = 0;
    for (int i = 0; i < batch_size; i++) {
      int tmp_real_h = static_cast<int>((cpu_shape_tensor.data<T>())[2 * i]);
      int tmp_real_w =
          static_cast<int>((cpu_shape_tensor.data<T>())[2 * i + 1]);
      if (tmp_real_h % out_stride[0] == 0) {
        tmp_real_h = tmp_real_h / out_stride[0];
      } else {
        tmp_real_h = tmp_real_h / out_stride[0] + 1;
      }
      if (tmp_real_w % out_stride[1] == 0) {
        tmp_real_w = tmp_real_w / out_stride[1];
      } else {
        tmp_real_w = tmp_real_w / out_stride[1] + 1;
      }
      img_real_h.push_back(tmp_real_h);
      img_real_w.push_back(tmp_real_w);
      output_height.push_back(Im2SeqOutputSize(
          img_real_h[i], kernels[0], paddings[0], paddings[2], strides[0]));
      output_width.push_back(Im2SeqOutputSize(
          img_real_w[i], kernels[1], paddings[1], paddings[3], strides[1]));
      result += output_height[i] * output_width[i];
    }

    out->Resize({result, img_channels * kernels[0] * kernels[1]});
    dev_ctx.template Alloc<T>(out);

    const std::vector<int> dilations({1, 1});
    int offset_out = 0;
    for (int i = 0; i < batch_size; i++) {
      const phi::DenseTensor src =
          in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
      phi::DenseTensor dst =
          out->Slice(offset_out,
                     offset_out + output_height[i] * output_width[i])
              .Resize({output_height[i],
                       output_width[i],
                       img_channels,
                       kernels[0],
                       kernels[1]});
      offset_out += output_height[i] * output_width[i];

      phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kOCF, Context, T> f;
      f(dev_ctx, src, dilations, strides, paddings, &dst);
    }
    phi::LegacyLoD lod(1);
    lod[0].reserve(batch_size + 1);
    int offset = 0;
    lod[0].push_back(offset);
    for (int i = 0; i < batch_size; ++i) {
      offset += output_height[i] * output_width[i];
      lod[0].push_back(offset);
    }
    out->set_lod(lod);
  } else {
    int output_height = Im2SeqOutputSize(
        img_height, kernels[0], paddings[0], paddings[2], strides[0]);
    int output_width = Im2SeqOutputSize(
        img_width, kernels[1], paddings[1], paddings[3], strides[1]);
    out->Resize(
        {static_cast<int64_t>(batch_size) * output_height * output_width,
         static_cast<int64_t>(img_channels) * kernels[0] * kernels[1]});
    dev_ctx.template Alloc<T>(out);
    const std::vector<int> dilations({1, 1});
    auto out_dims = out->dims();
    out->Resize({batch_size, out->numel() / batch_size});
    for (int i = 0; i < batch_size; i++) {
      const phi::DenseTensor src =
          in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
      phi::DenseTensor dst = out->Slice(i, i + 1).Resize(
          {output_height, output_width, img_channels, kernels[0], kernels[1]});

      phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kOCF, Context, T> f;
      f(dev_ctx, src, dilations, strides, paddings, &dst);
    }
    out->Resize(out_dims);
    phi::LegacyLoD lod(1);
    lod[0].reserve(batch_size + 1);
    int offset = 0;
    lod[0].push_back(offset);
    for (int i = 0; i < batch_size; ++i) {
      offset += output_height * output_width;
      lod[0].push_back(offset);
    }
    out->set_lod(lod);
  }
}

template <typename T, typename Context>
void Im2SequenceGradKernel(const Context& dev_ctx,
                           const DenseTensor& x_in,
                           const paddle::optional<DenseTensor>& y,
                           const DenseTensor& out_grad,
                           const std::vector<int>& kernels,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& out_stride,
                           DenseTensor* x_grad) {
  auto* in = &x_in;
  phi::DenseTensor tmp = out_grad;
  phi::DenseTensor* d_out = &tmp;
  auto* d_x = x_grad;
  dev_ctx.template Alloc<T>(d_x);

  auto x_v = phi::EigenVector<T>::Flatten(*d_x);
  auto& place = *dev_ctx.eigen_device();
  phi::funcs::EigenConstant<std::decay_t<decltype(place)>, T, 1>::Eval(
      place, x_v, 0.0);

  auto in_dim = in->dims();
  int batch_size = in_dim[0];
  int img_channels = in_dim[1];
  int img_height = in_dim[2];
  int img_width = in_dim[3];

  int output_height = Im2SeqOutputSize(
      img_height, kernels[0], paddings[0], paddings[2], strides[0]);
  int output_width = Im2SeqOutputSize(
      img_width, kernels[1], paddings[1], paddings[3], strides[1]);

  const std::vector<int> dilations({1, 1});

  auto d_out_dims = d_out->dims();
  d_out->Resize({batch_size, d_out->numel() / batch_size});
  for (int i = 0; i < batch_size; i++) {
    phi::DenseTensor dst =
        d_x->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
    const phi::DenseTensor src = d_out->Slice(i, i + 1).Resize(
        {output_height, output_width, img_channels, kernels[0], kernels[1]});
    phi::funcs::Col2ImFunctor<phi::funcs::ColFormat::kOCF, Context, T> f;
    f(dev_ctx, src, dilations, strides, paddings, &dst);
  }
  d_out->Resize(d_out_dims);
}
}  // namespace phi
