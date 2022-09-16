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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/transpose_grad_kernel.h"

namespace phi {
namespace funcs {

template <typename T>
class TransposeOneDNNHandler {
 public:
  TransposeOneDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                         std::vector<int>& axis,      // NOLINT
                         dnnl::engine engine)
      : dims_(dims),
        axis_(axis),
        logical_axis_(dims.size(), 0),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(
      const phi::funcs::OneDNNMemoryFormat& fmt, void* ptr) {
    // Make memory descriptor using input format, unless it
    // cannot be trusted (nchw) then make up memory fmt manually
    for (size_t i = 0; i < this->logical_axis_.size(); ++i) {
      this->logical_axis_[i] = i;
    }

    auto src_md = fmt != phi::funcs::OneDNNMemoryFormat::nchw
                      ? OneDNNMemDesc(dims_, oneDNNGetDataType<T>(), fmt)
                      : Axis2MemoryDesc(dims_, logical_axis_);
    return std::make_shared<dnnl::memory>(src_md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output,
                                                 phi::Place place) {
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
    dnnl::memory::desc mem_d(nchw_tz, oneDNNGetDataType<T>(), strides);

    return mem_d;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int> axis_;
  std::vector<int> logical_axis_;
  dnnl::engine engine_;
};
}  // namespace funcs

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  if (!x_grad) return;

  const auto& mkldnn_engine = dev_ctx.GetEngine();
  std::vector<int> reversed_axis(axis);
  int ndims = axis.size();
  if (ndims == 1) {
    phi::Copy(dev_ctx, out_grad, out_grad.place(), false, x_grad);
    x_grad->set_format(out_grad.format());
    return;
  }

  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }

  const T* out_grad_data = out_grad.data<T>();
  dev_ctx.template Alloc<T>(x_grad);
  auto nchw_tz = phi::vectorize<int64_t>(out_grad.dims());

  phi::funcs::TransposeOneDNNHandler<T> handler(
      nchw_tz, reversed_axis, mkldnn_engine);

  auto transpose_src_memory_p = handler.AcquireSrcMemory(
      out_grad.format(), phi::funcs::to_void_cast<T>(out_grad_data));
  auto transpose_dst_memory_p =
      handler.AcquireDstMemory(x_grad, dev_ctx.GetPlace());
  auto transpose_p =
      handler.AcquireTranspose(transpose_dst_memory_p, transpose_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  transpose_p->execute(
      astream, *transpose_src_memory_p, *transpose_dst_memory_p);
  astream.wait();
}

}  // namespace phi

PD_REGISTER_KERNEL(
    transpose2_grad, OneDNN, ALL_LAYOUT, phi::TransposeGradKernel, float) {}
