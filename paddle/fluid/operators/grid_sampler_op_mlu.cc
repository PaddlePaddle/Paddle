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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class GridSamplerMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_mlu_place(ctx.GetPlace()),
        true,
        platform::errors::Unavailable("This kernel only runs on MLU."));

    // input and output data
    const phi::DenseTensor* input = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* grid = ctx.Input<phi::DenseTensor>("Grid");
    phi::DenseTensor* output = ctx.Output<phi::DenseTensor>("Output");

    int n = input->dims()[0];
    int c = input->dims()[1];
    int out_h = grid->dims()[1];
    int out_w = grid->dims()[2];

    output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());

    // attrs
    // paddle.nn.functional.grid_sample(x, grid, mode='bilinear',
    // padding_mode='zeros', align_corners=True, name=None)
    const std::string mode = ctx.Attr<std::string>("mode");
    const std::string padding_mode = ctx.Attr<std::string>("padding_mode");
    bool align_corners = ctx.Attr<bool>("align_corners");
    const std::string data_format = phi::DataLayoutToString(input->layout());

    PADDLE_ENFORCE_EQ(
        mode == "bilinear",
        true,
        platform::errors::Unavailable(
            "Only support bilinear mode in mlu grid_sample kernel."));
    PADDLE_ENFORCE_EQ(
        padding_mode == "zeros",
        true,
        platform::errors::Unavailable(
            "Only support zeros padding_mode in mlu grid_sample kernel."));

    phi::DenseTensor trans_input(input->dtype());
    // transpose input from NCHW to NHWC
    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, input, &trans_input, true /*need_reshape_or_alloc*/);

    phi::DenseTensor tmp_output(output->dtype());
    tmp_output.mutable_data<T>({n, out_h, out_w, c}, ctx.GetPlace());

    MLUCnnlGridSampleDesc grid_sample_desc(mode, padding_mode, align_corners);
    MLUCnnlTensorDesc input_desc(
        trans_input, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc grid_desc(*grid, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc tmp_output_desc(
        tmp_output, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());

    MLUCnnl::GridSample(ctx,
                        grid_sample_desc.get(),
                        input_desc.get(),
                        GetBasePtr(&trans_input),
                        grid_desc.get(),
                        GetBasePtr(grid),
                        tmp_output_desc.get(),
                        GetBasePtr(&tmp_output));

    // transpose output from NHWC to NCHW
    const std::vector<int> perm_to_nchw = {
        0,
        3,
        1,
        2,
    };
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nchw,
                              &tmp_output,
                              output,
                              false /*need_reshape_or_alloc*/);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(grid_sampler,
                       ops::GridSamplerMLUKernel<float>,
                       ops::GridSamplerMLUKernel<plat::float16>);
