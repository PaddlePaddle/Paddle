/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using phi::DataLayout;
using phi::OneDNNContext;

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Transpose must use CPUPlace"));
    auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();
    std::vector<int> transpose_axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = transpose_axis.size();
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto& astream = OneDNNContext::tls().get_stream();

    if (ndims == 1) {
      framework::TensorCopy(*x, x->place(), out);
      out->set_mem_desc(x->mem_desc());
      return;
    }

    auto x_vec_dims = phi::vectorize(x->dims());

    auto x_type = phi::funcs::ToOneDNNDataType(x->dtype());
    phi::funcs::ReorderOneDNNHandler reorder_handler(
        x_vec_dims, x->dtype(), x_type, dnnl_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), phi::funcs::to_void_cast(x->data<T>()));

    auto dst_md =
        dnnl::memory::desc(x_vec_dims,
                           x->mem_desc().data_type(),
                           phi::funcs::GetPlainOneDNNFormat(x_vec_dims.size()));
    // a trick is used here to fake transpose of out_md, so later it will be
    // "untransposed", leaving output data in plain format tag
    auto dst_strides = FakeTranposeStrides(dst_md, transpose_axis);

    dst_md =
        dnnl::memory::desc(x_vec_dims, x->mem_desc().data_type(), dst_strides);
    auto dst_data =
        out->mutable_data(ctx.GetPlace(), x->type(), dst_md.get_size());

    auto reorder_dst_memory_p =
        std::make_shared<dnnl::memory>(dst_md, dnnl_engine, dst_data);

    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    out->set_mem_desc(reorder_dst_memory_p->get_desc().permute_axes(
        TransposeToPermuteAxis(transpose_axis)));
  }

 private:
  // it is needed because oneDNN's permute axis understand axes order in
  // different way PaddlePaddle's transpose
  std::vector<int> TransposeToPermuteAxis(
      const std::vector<int>& transpose_axis) const {
    std::vector<int> permute_axis(transpose_axis.size());

    for (size_t i = 0; i < transpose_axis.size(); ++i) {
      permute_axis[transpose_axis[i]] = i;
    }
    return permute_axis;
  }

  std::vector<int64_t> FakeTranposeStrides(
      const dnnl::memory::desc& dst_md,
      const std::vector<int>& transpose_axis) const {
    std::vector<int64_t> fake_strides(transpose_axis.size());
    auto dims = dst_md.dims();
    int total_stride = 1;
    int ndims = static_cast<int>(dims.size());

    for (int i = ndims - 1; i >= 0; --i) {
      fake_strides[transpose_axis[i]] = total_stride;
      total_stride *= dims[transpose_axis[i]];
    }

    return fake_strides;
  }
};

template <typename T>
class TransposeMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL TransposeGrad must use CPUPlace"));

    const auto* dout =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    if (!dx) return;
    auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();
    std::vector<int> transpose_axis = ctx.Attr<std::vector<int>>("axis");

    auto& astream = OneDNNContext::tls().get_stream();

    int ndims = transpose_axis.size();
    if (ndims == 1) {
      framework::TensorCopy(*dout, dout->place(), dx);
      dx->set_mem_desc(dout->mem_desc());
      return;
    }

    auto dout_vec_dims = phi::vectorize(dout->dims());
    auto dout_type = phi::funcs::ToOneDNNDataType(dout->dtype());

    phi::funcs::ReorderOneDNNHandler reorder_handler(
        dout_vec_dims, dout->dtype(), dout_type, dnnl_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->mem_desc(), phi::funcs::to_void_cast(dout->data<T>()));

    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(dx, dout->mem_desc(), ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();
    dx->set_mem_desc(
        reorder_dst_memory_p->get_desc().permute_axes(transpose_axis));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(transpose,
                   MKLDNN,
                   ::phi::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose_grad,
                   MKLDNN,
                   ::phi::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
