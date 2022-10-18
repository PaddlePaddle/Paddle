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

#include "paddle/phi/kernels/concat_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {
using memory = dnnl::memory;

namespace funcs {

template <typename T>
class ConcatOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::concat> {
 public:
  ConcatOneDNNHandler(Place cpu_place,
                      int concat_axis,
                      const dnnl::engine onednn_engine,
                      const std::vector<const DenseTensor*>& inputs,
                      DenseTensor* output)
      : OneDNNHandlerNoCachingT<T, dnnl::concat>(onednn_engine, cpu_place) {
    const int rank = inputs[0]->dims().size();

    PADDLE_ENFORCE_EQ(
        concat_axis >= -rank && concat_axis < rank,
        true,
        errors::InvalidArgument(
            "The axis is expected to be in range of [%d, %d), but got %d",
            -rank,
            rank,
            concat_axis));

    if (concat_axis < 0) {
      concat_axis = concat_axis + rank;
    }

    memory::data_type dt = ToOneDNNDataType(inputs[0]->dtype());
    std::vector<memory::desc> srcs_md;
    srcs_md.reserve(inputs.size());

    // Create memory descriptors for each of inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      srcs_md.push_back(inputs[i]->mem_desc());
    }

    auto dst_dims = vectorize<int64_t>(output->dims());

    memory::desc dst_md = memory::desc(dst_dims, dt, OneDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, concat_axis, srcs_md);
  }

  // (jczaja) concat oneDNN prim is not having .desc attribute so
  // we cannot use base AcquireForwardPrimitiveDescriptor
  void AcquireForwardPrimitiveDescriptor(
      const memory::desc& dst_md,
      const int concat_axis,
      const std::vector<memory::desc>& srcs_md) {
    this->fwd_pd_.reset(new dnnl::concat::primitive_desc(
        dst_md, concat_axis, srcs_md, this->engine_));
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor& input,
                                                 int i) {
    const T* input_data = input.data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }
};
}  // namespace funcs

static void EnforceLayouts(const std::vector<const DenseTensor*> inputs) {
  for (auto* input : inputs) {
    PADDLE_ENFORCE_EQ(
        input->layout(),
        DataLayout::ONEDNN,
        errors::InvalidArgument("Wrong layout set for Input tensor"));
  }
}

// From a multi-input, gather only nonempty inputs
static const std::vector<const DenseTensor*> ReduceMultiInput(
    const std::vector<const DenseTensor*>& inputs) {
  std::vector<const DenseTensor*> reduced(inputs.size());
  auto end_it = std::copy_if(
      inputs.begin(), inputs.end(), reduced.begin(), [](const DenseTensor* t) {
        return t->numel() > 0;
      });
  reduced.resize(std::distance(reduced.begin(), end_it));
  return reduced;
}

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const DenseTensor*>& x,
                  const Scalar& axis,
                  DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  // If any of the multiple inputs of concat has an input size of 0, the
  // actual size of the multi_input will change
  auto multi_input = ReduceMultiInput(x);
  EnforceLayouts(multi_input);

  auto out_dims_vec = vectorize(out->dims());
  if (std::any_of(out_dims_vec.begin(), out_dims_vec.end(), [](int64_t i) {
        return i < 0;
      })) {
    std::vector<phi::DDim> x_dims;
    x_dims.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      x_dims.push_back(x[i]->dims());
    }

    DDim out_dims =
        funcs::ComputeAndCheckShape(true, x_dims, axis.to<size_t>());
    out->Resize(out_dims);
  }

  funcs::ConcatOneDNNHandler<T> handler(
      dev_ctx.GetPlace(), axis.to<int>(), onednn_engine, multi_input, out);

  std::vector<std::shared_ptr<memory>> srcs;
  srcs.reserve(multi_input.size());

  auto dst_mem = handler.AcquireDstMemory(out);
  auto concat_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  std::unordered_map<int, memory> args;
  for (size_t i = 0; i < multi_input.size(); ++i) {
    srcs.push_back(handler.AcquireSrcMemory(*(multi_input[i]), i));
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, *(srcs.at(i))});
  }
  args.insert({DNNL_ARG_DST, *dst_mem});

  concat_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_mem->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(concat,
                   OneDNN,
                   ONEDNN,
                   phi::ConcatKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
