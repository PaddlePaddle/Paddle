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
    attrs.set_scales_mask(DNNL_ARG_SRC_0, /* mask = */ 0);

    src1_md_ = dnnl::memory::desc({1, sizeof(T)},
                                  OneDNNGetDataType<uint8_t>(),
                                  dnnl::memory::format_tag::ab);

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::algorithm::binary_add, src0_md, src1_md_, src0_md, attrs);
  }

  const dnnl::memory::desc& get_src1_md() const { return src1_md_; }

 private:
  dnnl::memory::desc src1_md_;
};

}  // namespace funcs

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  T fill_value = val.to<T>();
  out->Resize(common::make_ddim(shape.GetData()));

  funcs::FillConstantOneDNNHandler<T> handler(
      out, onednn_engine, dev_ctx.GetPlace());

  dnnl::memory constant_value_memory =
      dnnl::memory(handler.get_src1_md(),
                   onednn_engine,
                   reinterpret_cast<uint8_t*>(&fill_value));

  auto src0_memory_p = handler.AcquireDstMemory(out);
  auto fill_constant_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();

  std::vector<float> zero(1, 0);
  auto scales_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  auto scales = dnnl::memory(scales_md, onednn_engine, zero.data());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC_0, *src0_memory_p});
  args.insert({DNNL_ARG_SRC_1, constant_value_memory});
  args.insert({DNNL_ARG_DST, *src0_memory_p});
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scales});

  fill_constant_p->execute(astream, args);
  astream.wait();

  // src0_memory_p's md was just to allow the usage of a binary
  // primitive as a memset, and now we need to create a real one
  out->set_mem_desc({common::vectorize(out->dims()),
                     funcs::OneDNNGetDataType<T>(),
                     funcs::GetPlainOneDNNFormat(out->dims().size())});
}
}  // namespace phi

PD_REGISTER_KERNEL(full, OneDNN, ONEDNN, phi::FullKernel, float) {}
