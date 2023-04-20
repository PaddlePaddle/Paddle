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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_decl.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {
template <typename T, typename Context>
void TwoConv2dFusion(const Context& ctx,
                        const DenseTensor& input,
                        const DenseTensor& bias0,
                        const DenseTensor& bias1,
                        const DenseTensor& filter0,
                        const DenseTensor& filter1,
                        const std::vector<int>& strides,
                        const std::vector<int>& strides1,
                        const std::vector<int>& paddings,
                        const std::vector<int>& paddings1,
                        const std::string& padding_algorithm,
                        int groups,
                        const std::vector<int>& dilations,
                        const std::string& data_format,
                        const std::string& activation,
                        float fuse_alpha,
                        DenseTensor* output) {
   ctx.template Alloc<T>(output);
   auto in_dims = input.dims();
   auto filter0_dims = filter0.dims();
   auto filter1_dims = filter1.dims();
   auto out_dims = output->dims();
   int batch0 = in_dims[0];
   int ic0 = in_dims[3];
   int ih0 = in_dims[1];
   int iw0 = in_dims[2];
   int pad0_h0 = paddings[0];
   int pad0_h1 = paddings[0];
   int pad0_w0 = paddings[0];
   int pad0_w1 = paddings[0];
   int oc0 = filter0_dims[0];
   int kh0 = filter0_dims[1];
   int kw0 = filter0_dims[2];
   int stride0_h = strides[0];
   int stride0_w = strides[1];
   int dilation0_h = 1;
   int dilation0_w = 1;
   int oh0 = (ih0 + pad0_h0 + pad0_h1 - dilation0_h * (kh0 - 1) - 1) / stride0_h + 1;
   int ow0 = (iw0 + pad0_w0 + pad0_w1 - dilation0_w * (kw0 - 1) - 1) / stride0_w + 1;

   int batch1 = batch0;
   int ic1 = oc0;
   int ih1 = oh0;
   int iw1 = ow0;
   int pad1_h0 = paddings1[0];
   int pad1_h1 = paddings1[0];
   int pad1_w0 = paddings1[0];
   int pad1_w1 = paddings1[0];
   int oc1 = filter1_dims[0];
   int kh1 = filter1_dims[1];
   int kw1 = filter1_dims[2];
   int stride1_h = strides1[0];
   int stride1_w = strides1[1];
   int dilation1_h = 1;
   int dilation1_w = 1;
   int oh1 = out_dims[1];
   int ow1 = out_dims[2];

   ConvAllParams p0 = {reinterpret_cast<const half*>(input.data<T>()),
    reinterpret_cast<const half*>(filter0.data<T>()),
    reinterpret_cast<const half*>(bias0.data<T>()),
    nullptr,
    nullptr,
    batch0,
    ic0,
    ih0,
    iw0,
    kh0,
    kw0,
    oc0,
    pad0_h0,
    pad0_h1,
    pad0_w0,
    pad0_w1,
    stride0_h,
    stride0_w,
    dilation0_h,
    dilation0_w,
    oh0,
    ow0,
    1,
    &ctx};

    ConvAllParams p1 = {nullptr,
        reinterpret_cast<const half*>(filter1.data<T>()),
        reinterpret_cast<const half*>(bias1.data<T>()),
        nullptr,
        reinterpret_cast<half*>(output->data<T>()),
        batch1,
        ic1,
        ih1,
        iw1,
        kh1,
        kw1,
        oc1,
        pad1_h0,
        pad1_h1,
        pad1_w0,
        pad1_w1,
        stride1_h,
        stride1_w,
        dilation1_h,
        dilation1_w,
        oh1,
        ow1,
        1,
        &ctx};

        //TwoConv2dFusion(p0, p1);

}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(two_conv2d_fusion,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::TwoConv2dFusion,
                   float,
                   phi::dtype::float16) {}
