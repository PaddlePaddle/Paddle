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

#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using mkldnn::sum;
using paddle::framework::Tensor;
using paddle::platform::CPUDeviceContext;
using paddle::platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename T>
class SumMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    auto in_vars = ctx.MultiInputVar("X");

    const int N = in_vars.size();
    auto out_var = ctx.OutputVar("Out");
    bool in_place = out_var == in_vars[0];

    if (out_var->IsType<framework::LoDTensor>()) {
      LoDTensor* output = ctx.Output<LoDTensor>("Out");
      T* output_data = output->mutable_data<T>(ctx.GetPlace());

      std::vector<int> dst_tz = framework::vectorize2int(output->dims());
      auto src_tz = dst_tz;
      memory::format output_format{memory::format::format_undef};
      std::vector<float> scales;
      std::vector<memory::primitive_desc> srcs_mpd;
      std::vector<mkldnn::memory> srcs_mem;

      PADDLE_ENFORCE(in_vars[0]->IsType<LoDTensor>(),
                     "Input[0] must be LoDTensors");
      auto& input0 = in_vars[0]->Get<LoDTensor>();
      PADDLE_ENFORCE(input0.layout() == DataLayout::kMKLDNN &&
                         input0.format() != memory::format::format_undef,
                     "Wrong layout/format for inputs[0]");

      memory::format input_format = input0.format();

      for (int i = 0; i < N; i++) {
        PADDLE_ENFORCE(in_vars[i]->IsType<LoDTensor>(),
                       "all inputs must be all LoDTensors");
        auto& input = in_vars[i]->Get<LoDTensor>();
        PADDLE_ENFORCE(input.layout() == DataLayout::kMKLDNN &&
                           input.format() != memory::format::format_undef,
                       "Wrong layout/format for inputs");

        if (input.numel() == 0) {
          continue;
        }

        const T* input_data = input.data<T>();

        auto src_md =
            memory::desc(src_tz, memory::data_type::f32, input_format);
        auto src_mpd = memory::primitive_desc(src_md, mkldnn_engine);
        auto src_mem = memory(src_mpd, to_void_cast(input_data));
        srcs_mpd.push_back(src_mpd);
        srcs_mem.push_back(src_mem);
        scales.push_back(1.0);
      }

      auto dst_md =
          memory::desc(dst_tz, memory::data_type::f32, memory::format::any);

      auto sum_pd = sum::primitive_desc(dst_md, scales, srcs_mpd);

      std::shared_ptr<memory> dst_mem;
      if (in_place) {
        dst_mem.reset(new memory(sum_pd.dst_primitive_desc()));
      } else {
        dst_mem.reset(new memory(sum_pd.dst_primitive_desc(), output_data));
      }
      std::vector<mkldnn::primitive::at> inputs;
      for (size_t i = 0; i < srcs_mem.size(); ++i) {
        inputs.push_back(srcs_mem[i]);
      }

      auto sum_prim = mkldnn::sum(sum_pd, inputs, *dst_mem);
      output_format = (memory::format)platform::GetMKLDNNFormat(sum_pd);

      primitive reorder_prim;
      std::shared_ptr<memory> target_mem;
      if (in_place) {
        output_format = input_format;
        target_mem.reset(new memory(
            {{{src_tz}, memory::data_type::f32, output_format}, mkldnn_engine},
            output_data));
        reorder_prim = reorder(*dst_mem, *target_mem);
      }

      std::vector<primitive> pipeline;
      pipeline.push_back(sum_prim);
      if (in_place) pipeline.push_back(reorder_prim);
      stream(stream::kind::eager).submit(pipeline).wait();

      output->set_layout(DataLayout::kMKLDNN);
      output->set_format(output_format);
    } else {  // Fallback to naive version
      // TODO(@mozga-intel) Add MKLDNN SelectedRows & LoDTensorArray support
      SumKernel<CPUDeviceContext, T> reference_kernel;
      reference_kernel.Compute(ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(sum, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::SumMKLDNNOpKernel<float>);
