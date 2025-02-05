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

#include "paddle/phi/kernels/stack_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace funcs {
template <typename T>
class StackOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::concat> {
 public:
  StackOneDNNHandler(const Place& cpu_place,
                     int stack_axis,
                     const dnnl::engine onednn_engine,
                     const std::vector<const DenseTensor*>& inputs,
                     DenseTensor* output)
      : OneDNNHandlerNoCachingT<T, dnnl::concat>(onednn_engine, cpu_place) {
    int ndims = inputs[0]->dims().size();

    if (stack_axis < 0) {
      stack_axis = ndims + 1 + stack_axis;  // +1 to match output's ndims
    }

    // in stack op all inputs must have same dims
    auto input_dims = common::vectorize<int64_t>(inputs[0]->dims());

    dnnl::memory::data_type dt = ToOneDNNDataType(inputs[0]->dtype());
    std::vector<memory::desc> srcs_md;
    dnnl::memory::desc dst_md;
    OneDNNMemoryFormat dst_fmt;

    srcs_md.reserve(inputs.size());

    // if stack is not done on last(non existing) axis, then we can optimize
    // concat primitive by not adding additional dimension, since it causes
    // wrong output format deduction and suboptimal performance as a result
    if (stack_axis != ndims) {
      for (auto input : inputs) {
        srcs_md.push_back(input->mem_desc());
      }

      input_dims[stack_axis] *= inputs.size();  // NOLINT
      dst_md = dnnl::memory::desc(input_dims, dt, OneDNNMemoryFormat::any);
    } else {
      auto extended_input_dims = common::vectorize<int64_t>(output->dims());
      extended_input_dims[stack_axis] = 1;

      for (auto input : inputs) {
        srcs_md.push_back(input->mem_desc().reshape(extended_input_dims));
      }

      // concat primitive chooses suboptimal format tag because it cannot
      // distinguish between f.e. abcd and abdc if last dim is equal to 1 so
      // enforcing is needed for better performance
      dst_fmt = GetPlainOneDNNFormat(extended_input_dims.size());  // NOLINT
      dst_md =
          dnnl::memory::desc(common::vectorize(output->dims()), dt, dst_fmt);
    }

    this->AcquireForwardPrimitiveDescriptor(dst_md, stack_axis, srcs_md);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor& input,
                                                 int i) {
    const T* input_data = input.data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }
};
}  // namespace funcs

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& multi_input,
                 int axis,
                 DenseTensor* output) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  funcs::StackOneDNNHandler<T> handler(
      dev_ctx.GetPlace(), axis, onednn_engine, multi_input, output);

  std::vector<std::shared_ptr<dnnl::memory>> srcs;
  srcs.reserve(multi_input.size());

  auto dst_mem = handler.AcquireDstMemory(output);
  auto concat_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  std::unordered_map<int, dnnl::memory> args;
  for (size_t i = 0; i < multi_input.size(); ++i) {
    srcs.push_back(handler.AcquireSrcMemory(*(multi_input[i]), i));
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, *(srcs.at(i))});
  }
  args.insert({DNNL_ARG_DST, *dst_mem});

  concat_p->execute(astream, args);
  astream.wait();

  output->set_mem_desc(
      dst_mem->get_desc().reshape(common::vectorize(output->dims())));
}

}  // namespace phi

PD_REGISTER_KERNEL(stack, OneDNN, ONEDNN, phi::StackKernel, float) {}
