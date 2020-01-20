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
    auto out_var = ctx.OutputVar("Out");

    PADDLE_ENFORCE_NE(in_vars.empty(), true, platform::errors::InvalidArgument(
                                                 "Input variable is empty."));
    bool in_place = out_var == in_vars[0];

    LoDTensor* output = ctx.Output<LoDTensor>("Out");
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    auto dst_tz = framework::vectorize<int64_t>(output->dims());
    auto src_tz = dst_tz;
    MKLDNNMemoryFormat output_format{MKLDNNMemoryFormat::undef};
    std::vector<float> scales;
    std::vector<memory::desc> srcs_md;
    std::vector<mkldnn::memory> srcs_mem;

    auto& input0 = in_vars[0]->Get<LoDTensor>();
    in_place = (input0.numel() > 0) && (input0.data<T>() == output_data);

    MKLDNNMemoryFormat input_format = input0.format();

    for (size_t i = 0; i < in_vars.size(); i++) {
      auto& input_it = in_vars[i]->Get<LoDTensor>();
      if (input_it.numel() == 0) {
        continue;
      }

      const T* input_data = input_it.data<T>();

      auto src_md = memory::desc(src_tz, memory::data_type::f32, input_format);
      auto src_mem = memory(src_md, mkldnn_engine, to_void_cast(input_data));
      srcs_md.push_back(src_md);
      srcs_mem.push_back(src_mem);
      scales.push_back(1.0);
    }

    auto dst_md =
        memory::desc(dst_tz, memory::data_type::f32, MKLDNNMemoryFormat::any);

    auto sum_pd = sum::primitive_desc(dst_md, scales, srcs_md, mkldnn_engine);

    std::shared_ptr<memory> dst_mem;
    if (in_place) {
      dst_mem.reset(new memory(sum_pd.dst_desc(), mkldnn_engine));
    } else {
      dst_mem.reset(new memory(sum_pd.dst_desc(), mkldnn_engine, output_data));
    }

    auto sum_prim = mkldnn::sum(sum_pd);
    output_format = platform::GetMKLDNNFormat(sum_pd.dst_desc());

    std::shared_ptr<mkldnn::reorder> reorder_p;
    std::shared_ptr<memory> target_mem;
    if (in_place) {
      output_format = input_format;
      target_mem.reset(
          new memory({{src_tz}, memory::data_type::f32, output_format},
                     mkldnn_engine, output_data));
      reorder_p = std::make_shared<reorder>(*dst_mem, *target_mem);
    }

    mkldnn::stream astream(mkldnn_engine);
    std::unordered_map<int, memory> args;
    for (size_t i = 0; i < srcs_mem.size(); ++i) {
      args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, srcs_mem.at(i)});
    }
    args.insert({MKLDNN_ARG_DST, *dst_mem});

    sum_prim.execute(astream, args);
    astream.wait();

    if (in_place) {
      reorder_p->execute(astream, *dst_mem, *target_mem);
      astream.wait();
    }

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(output_format);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(sum, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::SumMKLDNNOpKernel<float>);
