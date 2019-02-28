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

#include <memory>
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::concat;
using mkldnn::stream;
using platform::to_void_cast;

static void EnforceLayouts(const std::vector<const Tensor*> inputs) {
  for (auto* input : inputs) {
    const bool is_layout_correct = input->layout() == DataLayout::kMKLDNN;
    const bool is_format_defined =
        input->format() != memory::format::format_undef;
    PADDLE_ENFORCE(is_layout_correct && is_format_defined,
                   "Wrong layout/format set for Input tensor");
  }
}

static memory::primitive_desc CreateMemPrimDesc(const Tensor& input,
                                                const mkldnn::engine& engine) {
  constexpr auto data_type = mkldnn::memory::f32;
  const auto dims = paddle::framework::vectorize2int(input.dims());
  const auto format = input.format();
  auto description = memory::desc(dims, data_type, format);
  auto mem_prim_desc = memory::primitive_desc(description, engine);
  return mem_prim_desc;
}

static platform::CPUPlace GetCpuPlace(
    const paddle::framework::ExecutionContext& ctx) {
  auto place = ctx.GetPlace();
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(place),
                 "It must use CPUPlace.");
  return boost::get<platform::CPUPlace>(place);
}

static const mkldnn::engine& GetMKLDNNEngine(
    const paddle::framework::ExecutionContext& ctx) {
  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
  return dev_ctx.GetEngine();
}

template <typename T>
class ConcatPrimitiveFactory {
 public:
  concat::primitive_desc CreateConcatPrimDescriptor(
      const std::vector<const Tensor*> multi_input, Tensor* output,
      int concat_axis, const mkldnn::engine& mkldnn_engine) {
    CreateSourcesDescriptors(multi_input, mkldnn_engine);
    auto dst_desc = CreateDstMemDescriptor(output);
    return concat::primitive_desc(dst_desc, concat_axis, srcs_pd);
  }

  concat CreateConcatPrimitive(const concat::primitive_desc& concat_pd,
                               Tensor* output, platform::CPUPlace place) {
    CreateSourcePrimitiveAts();
    dst_mem = CreateDstMemory(concat_pd, output, place);
    return concat(concat_pd, inputs, dst_mem.get());
  }

 private:
  memory::desc CreateDstMemDescriptor(Tensor* output) {
    auto dst_dims = paddle::framework::vectorize2int(output->dims());
    return memory::desc(dst_dims, platform::MKLDNNGetDataType<T>(),
                        memory::format::any);
  }

  mkldnn::memory CreateDstMemory(const concat::primitive_desc& concat_pd,
                                 Tensor* output, platform::CPUPlace place) {
    return memory(concat_pd.dst_primitive_desc(),
                  output->mutable_data<T>(place));
  }

  void CreateSourcesDescriptors(const std::vector<const Tensor*> multi_input,
                                const mkldnn::engine& mkldnn_engine) {
    for (size_t i = 0; i < multi_input.size(); i++) {
      auto mem_prim_desc = CreateMemPrimDesc(*multi_input[i], mkldnn_engine);
      srcs_pd.push_back(mem_prim_desc);
      srcs.push_back(
          memory(mem_prim_desc, to_void_cast(multi_input[i]->data<T>())));
    }
  }

  void CreateSourcePrimitiveAts() {
    inputs.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); i++) {
      inputs.push_back(srcs[i]);
    }
  }

 private:
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<memory> srcs;
  std::vector<primitive::at> inputs;
  boost::optional<memory> dst_mem;  // TODO(mgallus): change to std::optional
};                                  // upon introduction of C++17 to paddle

template <typename T>
class ConcatMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto place = GetCpuPlace(ctx);
    const auto& mkldnn_engine = GetMKLDNNEngine(ctx);

    auto multi_input = ctx.MultiInput<Tensor>("X");
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");
    int64_t concat_axis = static_cast<int64_t>(ctx.Attr<int>("axis"));

    ConcatPrimitiveFactory<T> prim_creator;
    auto concat_pd = prim_creator.CreateConcatPrimDescriptor(
        multi_input, output, static_cast<int>(concat_axis), mkldnn_engine);
    auto concat = prim_creator.CreateConcatPrimitive(concat_pd, output, place);
    stream(stream::kind::eager).submit({concat}).wait();

    output->set_mkldnn_prim_desc(concat_pd.dst_primitive_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(concat, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatMKLDNNOpKernel<float>)
