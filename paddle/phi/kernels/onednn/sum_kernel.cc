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
#include "paddle/phi/kernels/add_n_kernel.h"

namespace phi {
namespace funcs {
template <typename T>
class AddOneDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::sum> {
 public:
  AddOneDNNHandler(dnnl::engine engine,
                   const phi::Place& cpu_place,
                   const std::vector<const DenseTensor*>& x,
                   DenseTensor* z)

      : phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::sum>(engine, cpu_place),
        num_inputs_(0) {
    auto dst_tz = phi::vectorize<int64_t>(z->dims());
    auto src_tz = dst_tz;

    std::vector<dnnl::memory::desc> srcs_md;
    srcs_md.reserve(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      auto* input_it = x[i];
      if (input_it->numel() == 0) {
        continue;
      }
      srcs_md.push_back(input_it->mem_desc());
      ++num_inputs_;
    }
    std::vector<float> scales(num_inputs_, 1.0f);

    auto dst_md = dnnl::memory::desc(
        dst_tz, oneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, scales, srcs_md);
  }

  // (jczaja) sum oneDNN prim is not having .desc attribute so
  // we cannot use base AcquireForwardPrimitiveDescriptor
  void AcquireForwardPrimitiveDescriptor(
      const dnnl::memory::desc& dst_md,
      const std::vector<float>& scales,
      const std::vector<dnnl::memory::desc>& srcs_md) {
    this->fwd_pd_.reset(
        new dnnl::sum::primitive_desc(dst_md, scales, srcs_md, this->engine_));
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input,
                                                 int i) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }

  using phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::sum>::AcquireDstMemory;

  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc());
  }

  inline int GetNumInputs(void) { return num_inputs_; }

 private:
  int num_inputs_;
};
}  // namespace funcs

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const DenseTensor*>& x,
                DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      phi::errors::PreconditionNotMet("Operator DNNL Sum must use CPUPlace"));

  const auto& onednn_engine = dev_ctx.GetEngine();

  PADDLE_ENFORCE_NE(x.empty(),
                    true,
                    phi::errors::InvalidArgument("Input variable is empty."));
  auto* input0 = x[0];

  bool in_place = (input0->numel() > 0) && input0->IsSharedBufferWith(*out);

  funcs::AddOneDNNHandler<T> handler(onednn_engine, dev_ctx.GetPlace(), x, out);

  // Create list of SRC MEMs
  std::vector<std::shared_ptr<dnnl::memory>> srcs_mem;
  srcs_mem.reserve(handler.GetNumInputs());
  int input_index = 0;
  for (size_t i = 0; i < x.size(); i++) {
    auto* input_it = x[i];
    if (input_it->numel() == 0) {
      continue;
    }
    srcs_mem.push_back(handler.AcquireSrcMemory(input_it, input_index));
    ++input_index;
  }

  std::unordered_map<int, dnnl::memory> args;
  std::shared_ptr<dnnl::memory> dst_mem;

  for (size_t i = 0; i < srcs_mem.size(); ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, *(srcs_mem[i])});
  }

  if (in_place) {
    dst_mem = srcs_mem[0];
  } else {
    dst_mem = handler.AcquireDstMemory(out);
  }
  args.insert({DNNL_ARG_DST, *dst_mem});

  auto sum_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  sum_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_mem->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(shape,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
