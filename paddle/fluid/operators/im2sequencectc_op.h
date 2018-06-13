/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include <fstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

inline int Im2SeqOutputSize(int input_size, int filter_size, int padding_0,
                            int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

template <typename DeviceContext, typename T>
class Im2SequenceCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* in = ctx.Input<Tensor>("X");
    // TODO(fuhailong): add new data layer to solve multibatch inference
    const Tensor* imgRealSize = ctx.Input<Tensor>("Y");
    LoDTensor* out = ctx.Output<LoDTensor>("Out");
    auto in_dim = in->dims();
    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];
    float* imgreal_size =
        reinterpret_cast<float*>(malloc(batch_size * 2 * sizeof(float)));
    // TODO(fuhailong): fix gpu inference
    Tensor cpu_shape_tensor;
    TensorCopySync(*imgRealSize, platform::CPUPlace(), &cpu_shape_tensor);
    memcpy(imgreal_size, cpu_shape_tensor.data<T>(),
           sizeof(float) * 2 * batch_size);
    auto imgRealSize_dim = imgRealSize->dims();
    auto kernels = ctx.Attr<std::vector<int>>("kernels");
    auto strides = ctx.Attr<std::vector<int>>("strides");
    auto paddings = ctx.Attr<std::vector<int>>("paddings");
    auto out_stride = ctx.Attr<std::vector<int>>("out_stride");
    auto is_inference = ctx.Attr<bool>("is_inference");
    PADDLE_ENFORCE_EQ(is_inference, 1, "im2sequencectc_op use to inference");
    if (batch_size > 1) {
      std::vector<int> imgReal_H;
      std::vector<int> imgReal_W;
      std::vector<int> output_height;
      std::vector<int> output_width;
      int result = 0;
      for (int i = 0; i < batch_size; i++) {
        int tmp_real_H = static_cast<int>(imgreal_size[2 * i]);
        int tmp_real_W = static_cast<int>(imgreal_size[2 * i + 1]);
        if (tmp_real_H % out_stride[0] == 0) {
          tmp_real_H = tmp_real_H / out_stride[0];
        } else {
          tmp_real_H = tmp_real_H / out_stride[0] + 1;
        }
        if (tmp_real_W % out_stride[1] == 0) {
          tmp_real_W = tmp_real_W / out_stride[1];
        } else {
          tmp_real_W = tmp_real_W / out_stride[1] + 1;
        }
        imgReal_H.push_back(tmp_real_H);
        imgReal_W.push_back(tmp_real_W);
        output_height.push_back(Im2SeqOutputSize(
            imgReal_H[i], kernels[0], paddings[0], paddings[2], strides[0]));
        output_width.push_back(Im2SeqOutputSize(
            imgReal_W[i], kernels[1], paddings[1], paddings[3], strides[1]));
        // TODO(fuhailong): compute dims of output
        // call: out->mutable_data<T>(ctx.GetPlace(), output_dims);
        result += output_height[i] * output_width[i];
      }

      out->mutable_data<T>({result, img_channels * kernels[0] * kernels[1]},
                           ctx.GetPlace());

      const std::vector<int> dilations({1, 1});
      // TODO(fuhailong): out_dims has two index,
      // out_dims[0] and out_dims[1],
      // {batchsize*output_height*output_width,channel*kernel[0],*kernel[1]},
      // multi batch ,the first place is output_height[i]*output_width[i].
      int offset_out = 0;
      for (int i = 0; i < batch_size; i++) {
        const Tensor src =
            in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
        // TODO(fuhailong): add image real size
        Tensor dst = out->Slice(offset_out,
                                offset_out + output_height[i] * output_width[i])
                         .Resize({output_height[i], output_width[i],
                                  img_channels, kernels[0], kernels[1]});
        offset_out += output_height[i] * output_width[i];

        math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
        // eq, kOCF cnn to rnn format
        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        f(dev_ctx, src, dilations, strides, paddings, &dst);
      }
      // set lod information
      // TODO(wanghaoshuang): Move this to InferShape
      framework::LoD lod(1);
      lod[0].reserve(batch_size + 1);
      int offset = 0;
      lod[0].push_back(offset);
      for (int i = 0; i < batch_size; ++i) {
        offset += output_height[i] * output_width[i];
        lod[0].push_back(offset);
      }
      out->set_lod(lod);
    } else if (batch_size == 1) {
      out->mutable_data<T>(ctx.GetPlace());
      int output_height = Im2SeqOutputSize(img_height, kernels[0], paddings[0],
                                           paddings[2], strides[0]);
      int output_width = Im2SeqOutputSize(img_width, kernels[1], paddings[1],
                                          paddings[3], strides[1]);

      const std::vector<int> dilations({1, 1});
      auto out_dims = out->dims();
      out->Resize({batch_size, out->numel() / batch_size});
      for (int i = 0; i < batch_size; i++) {
        const Tensor src =
            in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
        Tensor dst =
            out->Slice(i, i + 1).Resize({output_height, output_width,
                                         img_channels, kernels[0], kernels[1]});

        math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        f(dev_ctx, src, dilations, strides, paddings, &dst);
      }
      out->Resize(out_dims);
      // set lod information
      // TODO(wanghaoshuang): Move this to InferShape
      framework::LoD lod(1);
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
};

}  // namespace operators
}  // namespace paddle
