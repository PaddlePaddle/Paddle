//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*Licensed under the Apache License, Version 2.0(the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
namespace platform {
class CPUDeviceContext;
class MKLDNNDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using paddle::platform::CPUDeviceContext;
using paddle::platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename T>
class SumMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::sum> {
 public:
  SumMKLDNNHandler(mkldnn::engine engine, platform::Place cpu_place,
                   const std::vector<framework::Variable*>& in_vars,
                   framework::LoDTensor* z)

      : platform::MKLDNNHandlerNoCachingT<T, dnnl::sum>(engine, cpu_place),
        num_inputs_(0) {
    auto dst_tz = framework::vectorize<int64_t>(z->dims());
    auto src_tz = dst_tz;

    std::vector<mkldnn::memory::desc> srcs_md;
    for (size_t i = 0; i < in_vars.size(); ++i) {
      auto& input_it = in_vars[i]->Get<framework::LoDTensor>();
      if (input_it.numel() == 0) {
        continue;
      }
      srcs_md.push_back(input_it.mem_desc());
      ++num_inputs_;
    }
    std::vector<float> scales(num_inputs_, 1.0f);

    auto dst_md = mkldnn::memory::desc(dst_tz, platform::MKLDNNGetDataType<T>(),
                                       MKLDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, scales, srcs_md);
  }

  // (jczaja) sum oneDNN prim is not having .desc attribute so
  // we cannot use base AcquireForwardPrimitiveDescriptor
  void AcquireForwardPrimitiveDescriptor(
      const mkldnn::memory::desc& dst_md, const std::vector<float>& scales,
      const std::vector<mkldnn::memory::desc>& srcs_md) {
    this->fwd_pd_.reset(
        new dnnl::sum::primitive_desc(dst_md, scales, srcs_md, this->engine_));
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const framework::Tensor& input, int i) {
    const T* input_data = input.data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }

  using platform::MKLDNNHandlerNoCachingT<T, dnnl::sum>::AcquireDstMemory;

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc());
  }

  inline int GetNumInputs(void) { return num_inputs_; }

 private:
  int num_inputs_;
};

template <typename T>
class SumMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Sum must use CPUPlace"));
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    auto in_vars = ctx.MultiInputVar("X");

    PADDLE_ENFORCE_NE(in_vars.empty(), true, platform::errors::InvalidArgument(
                                                 "Input variable is empty."));
    auto& input0 = in_vars[0]->Get<LoDTensor>();
    LoDTensor* output = ctx.Output<LoDTensor>("Out");

    bool in_place = (input0.numel() > 0) && input0.IsSharedBufferWith(*output);

    SumMKLDNNHandler<T> handler(mkldnn_engine, ctx.GetPlace(), in_vars, output);

    // Create list of SRC MEMs
    std::vector<std::shared_ptr<mkldnn::memory>> srcs_mem;
    srcs_mem.reserve(handler.GetNumInputs());
    int input_index = 0;
    for (size_t i = 0; i < in_vars.size(); ++i) {
      auto& input_it = in_vars[i]->Get<framework::LoDTensor>();
      if (input_it.numel() == 0) {
        continue;
      }
      srcs_mem.push_back(handler.AcquireSrcMemory(input_it, input_index));
      ++input_index;
    }

    auto dst_mem = in_place ? handler.AcquireDstMemory()
                            : handler.AcquireDstMemory(output);

    auto sum_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, mkldnn::memory> args;
    for (size_t i = 0; i < srcs_mem.size(); ++i) {
      args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, *(srcs_mem[i])});
    }
    args.insert({MKLDNN_ARG_DST, *dst_mem});

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    sum_p->execute(astream, args);
    astream.wait();

    output->set_layout(framework::DataLayout::kMKLDNN);
    output->set_mem_desc(dst_mem->get_desc());

    // For in-place execution which sum does not have we need to fake it
    // so from oneDNN dst memory we reorder data into input
    if (in_place) {
      auto& in_out = in_vars[0]->Get<framework::LoDTensor>();
      auto output_tz = framework::vectorize<int64_t>(output->dims());
      platform::ReorderMKLDNNHandler reorder_handler(
          output_tz, output->type(), framework::ToMKLDNNDataType(in_out.type()),
          dev_ctx.GetEngine());

      auto target_mem = reorder_handler.AcquireDstMemory(
          output, output->mem_desc(), ctx.GetPlace());

      auto reorder_p = reorder_handler.AcquireReorder(target_mem, dst_mem);
      {
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, *dst_mem, *target_mem);
        astream.wait();
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(
    sum, MKLDNN, ::paddle::platform::CPUPlace,
    paddle::operators::SumMKLDNNOpKernel<paddle::platform::bfloat16>,
    paddle::operators::SumMKLDNNOpKernel<float>);
