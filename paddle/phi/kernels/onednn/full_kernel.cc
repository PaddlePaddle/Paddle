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

#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace funcs {

template <typename T>
class FillConstantOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  FillConstantOneDNNHandler(DenseTensor* out,
                            dnnl::engine engine,
                            Place cpu_place)
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src0_md = dnnl::memory::desc({out->numel(), sizeof(T)},
                                            OneDNNGetDataType<uint8_t>(),
                                            dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attrs;
    attrs.set_scales(DNNL_ARG_SRC_0, /* mask = */ 0, {0.0f});

    this->AcquireForwardPrimitiveDescriptor(
        attrs, dnnl::algorithm::binary_add, src0_md, src1_md, src0_md);
  }

  static const dnnl::memory::desc src1_md;
};

template <typename T>
const dnnl::memory::desc FillConstantOneDNNHandler<T>::src1_md(
    {1, sizeof(T)}, OneDNNGetDataType<uint8_t>(), dnnl::memory::format_tag::ab);
}  // namespace funcs

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  T fill_value = val.to<T>();
  out->Resize(make_ddim(shape.GetData()));

  funcs::FillConstantOneDNNHandler<T> handler(
      out, onednn_engine, dev_ctx.GetPlace());

  dnnl::memory constant_value_memory =
      dnnl::memory(funcs::FillConstantOneDNNHandler<T>::src1_md,
                   onednn_engine,
                   reinterpret_cast<uint8_t*>(&fill_value));

  auto src0_memory_p = handler.AcquireDstMemory(out);
  auto fill_constant_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  fill_constant_p->execute(astream,
                           {{DNNL_ARG_SRC_0, *src0_memory_p},
                            {DNNL_ARG_SRC_1, constant_value_memory},
                            {DNNL_ARG_DST, *src0_memory_p}});
  astream.wait();

  // src0_memory_p's md was just to allow the usage of a binary
  // primitive as a memset, and now we need to create a real one
  out->set_mem_desc({vectorize(out->dims()),
                     funcs::OneDNNGetDataType<T>(),
                     funcs::GetPlainOneDNNFormat(out->dims().size())});
}
}  // namespace phi

PD_REGISTER_KERNEL(full, OneDNN, ALL_LAYOUT, phi::FullKernel, float) {}
