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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using framework::DataLayout;

template <typename T>
class TransposeMKLDNNHandler {
 public:
  TransposeMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                         std::vector<int>& axis,      // NOLINT
                         dnnl::engine engine)
      : dims_(dims),
        axis_(axis),
        logical_axis_(dims.size(), 0),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const MKLDNNMemoryFormat& fmt,
                                                 void* ptr) {
    // Make memory descriptor using input format, unless it
    // cannot be trusted (nchw) then make up memory fmt manually
    for (size_t i = 0; i < this->logical_axis_.size(); ++i) {
      this->logical_axis_[i] = i;
    }

    auto src_md = fmt != MKLDNNMemoryFormat::nchw
                      ? platform::MKLDNNMemDesc(
                            dims_, platform::MKLDNNGetDataType<T>(), fmt)
                      : Axis2MemoryDesc(dims_, logical_axis_);
    return std::make_shared<dnnl::memory>(src_md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor* output,
                                                 platform::Place place) {
    auto dst_md = Axis2MemoryDesc(dims_, axis_);
    auto dst_data = output->mutable_data<T>(place, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::reorder> AcquireTranspose(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    return std::make_shared<dnnl::reorder>(*(src_memory_p), *(dst_memory_p));
  }

 protected:
  dnnl::memory::desc Axis2MemoryDesc(std::vector<int64_t>& nchw_tz,  // NOLINT
                                     std::vector<int>& axis          // NOLINT
  ) {
    size_t ndims = axis.size();

    std::vector<int64_t> strides(ndims);
    unsigned int total_stride = 1;
    for (int i = ndims - 1; i >= 0; --i) {
      strides[axis[i]] = total_stride;
      total_stride *= nchw_tz[axis[i]];
    }
    dnnl::memory::desc mem_d(
        nchw_tz, platform::MKLDNNGetDataType<T>(), strides);

    return mem_d;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int> axis_;
  std::vector<int> logical_axis_;
  dnnl::engine engine_;
};

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Transpose must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    const T* input_data = input->data<T>();

    if (ndims == 1) {
      framework::TensorCopy(*input, input->place(), output);
      output->set_format(input->format());
      return;
    }

    auto nchw_tz = phi::vectorize<int64_t>(input->dims());

    TransposeMKLDNNHandler<T> handler(nchw_tz, axis, mkldnn_engine);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        input->format(), platform::to_void_cast<T>(input_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(output, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    transpose_p->execute(
        astream, *transpose_src_memory_p, *transpose_dst_memory_p);
    astream.wait();

    output->set_layout(DataLayout::kNCHW);
    output->set_format(MKLDNNMemoryFormat::undef);
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
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    if (!x_grad) return;
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    int ndims = axis.size();
    if (ndims == 1) {
      framework::TensorCopy(*out_grad, out_grad->place(), x_grad);
      x_grad->set_format(out_grad->format());
      return;
    }

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    const T* out_grad_data = out_grad->data<T>();
    x_grad->mutable_data<T>(ctx.GetPlace());

    auto nchw_tz = phi::vectorize<int64_t>(out_grad->dims());

    TransposeMKLDNNHandler<T> handler(nchw_tz, reversed_axis, mkldnn_engine);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        out_grad->format(), platform::to_void_cast<T>(out_grad_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(x_grad, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    transpose_p->execute(
        astream, *transpose_src_memory_p, *transpose_dst_memory_p);
    astream.wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kTransposeMKLDNNFP32,
                                    ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8,
                                    ops::kTransposeMKLDNNINT8,
                                    ops::TransposeMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8,
                                    ops::kTransposeMKLDNNINT8,
                                    ops::TransposeMKLDNNOpKernel<int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    transpose2,
    MKLDNN,
    ::paddle::platform::CPUPlace,
    BF16,
    ops::kTransposeMKLDNNFP32,
    ops::TransposeMKLDNNOpKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(transpose,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose_grad,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);

REGISTER_OP_KERNEL(transpose2_grad,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
