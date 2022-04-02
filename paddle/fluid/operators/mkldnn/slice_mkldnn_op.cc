/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

static dnnl::memory::format_tag get_plain_format_tag(
    const paddle::framework::Tensor* tensor) {
  auto tensor_dims_size = tensor->dims().size();

  switch (tensor_dims_size) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
  }

  return dnnl::memory::format_tag::abcdef;
}

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class SliceMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("Input");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_vec_dims = phi::vectorize(x->dims());

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");

    std::vector<int64_t> axes(ctx.Attr<std::vector<int>>("axes").begin(),
                              ctx.Attr<std::vector<int>>("axes").end());
    std::vector<int64_t> starts(ctx.Attr<std::vector<int>>("starts").begin(),
                                ctx.Attr<std::vector<int>>("starts").end());
    std::vector<int64_t> ends(ctx.Attr<std::vector<int>>("ends").begin(),
                              ctx.Attr<std::vector<int>>("ends").end());

    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }

    std::vector<int64_t> offsets(x_vec_dims.size(), 0);
    std::vector<int64_t> slice_dims(x_vec_dims);

    for (size_t i = 0; i < axes.size(); ++i) {
      starts[i] = starts[i] < 0 ? x_vec_dims[axes[i]] + starts[i] : starts[i];
      ends[i] = ends[i] < 0 ? x_vec_dims[axes[i]] + ends[i]
                            : std::min(ends[i], x_vec_dims[axes[i]]);
      offsets[axes[i]] = starts[i];
      slice_dims[axes[i]] = ends[i] - starts[i];
    }

    out->Resize(phi::make_ddim(slice_dims));

    dnnl::memory::data_type x_type =
        framework::ToMKLDNNDataType(framework::TransToProtoVarType(x->dtype()));

    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, framework::TransToProtoVarType(x->dtype()), x_type,
        onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->format(), platform::to_void_cast(x->data<T>()));
    auto slice_mem_p = reorder_handler.AcquireSubmemory(slice_dims, offsets,
                                                        reorder_src_memory_p);
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, slice_dims, get_plain_format_tag(x), ctx.GetPlace());

    auto reorder_p =
        reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

    std::vector<int64_t> new_out_dims(slice_dims.size() - decrease_axis.size());

    if (new_out_dims.size() == 0) {
      new_out_dims.emplace_back(1);
    } else {
      for (const auto& axis : decrease_axis) {
        slice_dims[axis] = 0;
      }

      int i = 0;
      for (const auto& slice_dim : slice_dims) {
        if (slice_dim != 0) new_out_dims[i++] = slice_dim;
      }
    }

    astream.wait();
    out->Resize(phi::make_ddim(new_out_dims));
    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(
        reorder_dst_memory_p->get_desc().reshape(new_out_dims)));
  }
};
template <typename T>
class SliceGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));

    auto dx_vec_dims = phi::vectorize(dx->dims());
    auto dout_vec_dims = phi::vectorize(dout->dims());

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");

    std::vector<int64_t> axes(ctx.Attr<std::vector<int>>("axes").begin(),
                              ctx.Attr<std::vector<int>>("axes").end());
    std::vector<int64_t> starts(ctx.Attr<std::vector<int>>("starts").begin(),
                                ctx.Attr<std::vector<int>>("starts").end());
    std::vector<int64_t> ends(ctx.Attr<std::vector<int>>("ends").begin(),
                              ctx.Attr<std::vector<int>>("ends").end());

    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    std::vector<int64_t> offsets(dx_vec_dims.size(), 0);
    std::vector<int64_t> slice_dims(dx_vec_dims);

    for (size_t i = 0; i < axes.size(); ++i) {
      starts[i] = starts[i] < 0 ? dx_vec_dims[axes[i]] + starts[i] : starts[i];
      ends[i] = ends[i] < 0 ? dx_vec_dims[axes[i]] + ends[i]
                            : std::min(ends[i], dx_vec_dims[axes[i]]);
      offsets[axes[i]] = starts[i];
      slice_dims[axes[i]] = ends[i] - starts[i];
    }

    dnnl::memory::data_type dout_type = framework::ToMKLDNNDataType(
        framework::TransToProtoVarType(dout->dtype()));
    dnnl::memory::desc md(dout_vec_dims, platform::MKLDNNGetDataType<T>(),
                          dout->format());
    dnnl::memory::format_tag reorder_format_tag =
        platform::GetMKLDNNFormat(md.reshape(slice_dims));

    platform::ReorderMKLDNNHandler reorder_handler(
        slice_dims, framework::TransToProtoVarType(dout->dtype()), dout_type,
        onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        reorder_format_tag, platform::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        dx, dx_vec_dims, reorder_format_tag, ctx.GetPlace());
    memset(dx->data<T>(), 0, reorder_dst_memory_p->get_desc().get_size());

    auto slice_mem_p = reorder_handler.AcquireSubmemory(slice_dims, offsets,
                                                        reorder_dst_memory_p);

    auto reorder_p =
        reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
    astream.wait();

    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(reorder_format_tag);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(slice, MKLDNN, paddle::platform::CPUPlace,
                   ops::SliceMKLDNNKernel<float>,
                   ops::SliceMKLDNNKernel<int8_t>,
                   ops::SliceMKLDNNKernel<uint8_t>,
                   ops::SliceMKLDNNKernel<paddle::platform::bfloat16>);

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(slice_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::SliceGradMKLDNNKernel<float>,
                   ops::SliceGradMKLDNNKernel<paddle::platform::bfloat16>);
