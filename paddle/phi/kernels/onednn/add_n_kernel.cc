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

#include "paddle/phi/kernels/add_n_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
bool AddNCheckIfOneDNNSupport(const KernelContext* ctx) {
  for (size_t i = 0; i < ctx->InputsSize(); i++) {
    if (!DenseTensor::classof(ctx->MutableIutputAt(i))) {
      return false;
    }
  }
  KernelContext* ctx_tmp = const_cast<KernelContext*>(ctx);
  if (!DenseTensor::classof(ctx_tmp->MutableOutputAt(0))) {
    return false;
  }
  return true;
}

namespace funcs {
template <typename T>
class SumOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::sum> {
 public:
  SumOneDNNHandler(dnnl::engine engine,
                   const Place& cpu_place,
                   const std::vector<const TensorBase*>& x,
                   DenseTensor* out)

      : OneDNNHandlerNoCachingT<T, dnnl::sum>(engine, cpu_place),
        num_inputs_(0) {
    auto dst_tz = common::vectorize<int64_t>(out->dims());
    auto src_tz = dst_tz;

    std::vector<dnnl::memory::desc> srcs_md;
    srcs_md.reserve(x.size());
    for (auto item : x) {
      auto* input_it = (static_cast<const DenseTensor*>(item));
      if (input_it->numel() == 0) {
        continue;
      }
      srcs_md.push_back(input_it->mem_desc());
      ++num_inputs_;
    }
    std::vector<float> scales(num_inputs_, 1.0f);

    auto dst_md = dnnl::memory::desc(
        dst_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, scales, srcs_md);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input,
                                                 int i) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }

  using OneDNNHandlerNoCachingT<T, dnnl::sum>::AcquireDstMemory;

  std::shared_ptr<dnnl::memory> AcquireDstMemory() {
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc());
  }

  inline int GetNumInputs() { return num_inputs_; }

 private:
  int num_inputs_;
};
}  // namespace funcs

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const TensorBase*>& x,
                DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == AllocationType::CPU,
      true,
      errors::PreconditionNotMet("oneDNN AddN kernel must use CPUPlace"));

  const auto& onednn_engine = dev_ctx.GetEngine();

  PADDLE_ENFORCE_NE(
      x.empty(), true, errors::InvalidArgument("Input variable is empty."));
  auto* input0 = (static_cast<const DenseTensor*>(x[0]));

  bool in_place = (input0->numel() > 0) && input0->IsSharedBufferWith(*out);

  funcs::SumOneDNNHandler<T> handler(onednn_engine, dev_ctx.GetPlace(), x, out);

  // Create list of SRC MEMs
  std::vector<std::shared_ptr<dnnl::memory>> srcs_mem;
  srcs_mem.reserve(handler.GetNumInputs());
  int input_index = 0;
  for (auto item : x) {
    auto* input_it = (static_cast<const DenseTensor*>(item));
    if (input_it->numel() == 0) {
      continue;
    }
    srcs_mem.push_back(handler.AcquireSrcMemory(input_it, input_index));
    ++input_index;
  }

  std::unordered_map<int, dnnl::memory> args;

  for (size_t i = 0; i < srcs_mem.size(); ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, *(srcs_mem[i])});
  }

  auto dst_mem = in_place ? srcs_mem[0] : handler.AcquireDstMemory(out);

  args.insert({DNNL_ARG_DST, *dst_mem});

  auto sum_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  sum_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_mem->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(
    add_n, OneDNN, ONEDNN, phi::AddNKernel, float, phi::dtype::bfloat16) {
  kernel->check_if_onednn_kernel_support_ = phi::AddNCheckIfOneDNNSupport;
}
