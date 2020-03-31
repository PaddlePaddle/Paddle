/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "dnnl.hpp"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
// #include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
// using dnnl::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
// using dnnl::stream;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

template <typename T>
class DNNLMatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    auto dim_x = math::CreateMatrixDescriptor(RowMatrixFromVector(x->dims()), 0,
                                              ctx.Attr<bool>("transpose_X"));
    auto dim_y = math::CreateMatrixDescriptor(ColumnMatrixFromVector(y->dims()),
                                              0, ctx.Attr<bool>("transpose_Y"));

    memory::dim batch_size = 0;
    if (dim_x.batch_size_ == 0 && dim_y.batch_size_ == 0) {
      batch_size = 1;
    } else {
      PADDLE_ENFORCE(
          dim_x.batch_size_ == dim_y.batch_size_ || dim_x.batch_size_ == 0 ||
              dim_y.batch_size_ == 0,
          "dim_x.batch_size should be equal to dim_y.batch_size, or "
          "one of dim_x.batch_size and dim_y.batch_size should be 0. "
          "But got dim_x.batch_size = %d, dim_y.batch_size = %d.",
          dim_x.batch_size_, dim_y.batch_size_);
      batch_size = dim_x.batch_size_;
    }
    const memory::dim M = dim_x.height_;
    const memory::dim N = dim_y.width_;
    const memory::dim K = dim_x.width_;

    memory::dims src_dims = {batch_size, M, K};
    memory::dims weights_dims = {batch_size, K, N};
    memory::dims dst_dims = {batch_size, M, N};

    // Translate transA and transB
    memory::dims x_strides =
        !dim_x.trans_ ? memory::dims{M * K, K, 1} : memory::dims{M * K, 1, M};
    memory::dims y_strides =
        !dim_y.trans_ ? memory::dims{N * K, N, 1} : memory::dims{N * K, 1, K};
    // memory::dims out_strides = memory::dims{M * N, ldout, 1};

    // Create memory descriptors and memory objects for src, weights and
    // dst.
    auto src_md = memory::desc(src_dims, MKLDNNGetDataType<T>(), x_strides);
    auto weights_md =
        memory::desc(weights_dims, MKLDNNGetDataType<T>(), y_strides);
    auto src_mem = dnnl::memory(src_md, engine, to_void_cast(x->data<T>()));
    auto weights_mem =
        dnnl::memory(weights_md, engine, to_void_cast(y->data<T>()));

    bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
    memory::desc dst_md;
    dnnl::memory dst_mem;
    if (force_fp32_out) {
      dst_md = memory::desc(dst_dims, memory::data_type::f32,
                            memory::format_tag::abc);
      dst_mem =
          dnnl::memory(dst_md, engine,
                       to_void_cast(out->mutable_data<float>(ctx.GetPlace())));
    } else {
      dst_md = memory::desc(dst_dims, MKLDNNGetDataType<T>(),
                            memory::format_tag::abc);
      dst_mem = dnnl::memory(
          dst_md, engine, to_void_cast(out->mutable_data<T>(ctx.GetPlace())));
    }

    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
    float out_shift_scale = scale_out / (scale_x * scale_y);
    float alpha = ctx.Attr<float>("alpha");
    float final_scale_out = out_shift_scale * alpha;
    dnnl::primitive_attr attr;
    if (final_scale_out != 1.0f)
      attr.set_output_scales(/* mask */ 0, {final_scale_out});

    auto matmul_d = dnnl::matmul::desc(src_md, weights_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    dnnl::stream stream(engine);
    matmul_prim.execute(stream, {
                                    {MKLDNN_ARG_SRC, src_mem},
                                    {MKLDNN_ARG_WEIGHTS, weights_mem},
                                    {MKLDNN_ARG_DST, dst_mem},
                                });
    stream.wait();

    out->set_layout(DataLayout::kMKLDNN);
    out->set_format(platform::MKLDNNFormatForSize(out->dims().size(),
                                                  MKLDNNMemoryFormat::nchw));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::DNNLMatMulKernel<float>, ops::DNNLMatMulKernel<int8_t>,
                   ops::DNNLMatMulKernel<uint8_t>);
