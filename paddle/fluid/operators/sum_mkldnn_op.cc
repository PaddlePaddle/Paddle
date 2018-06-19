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

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::CPUDeviceContext;
using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::sum;
using mkldnn::reorder;
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

      if (src_tz.size() == 1 && (input_format == memory::format::nchw ||
                                 input_format == memory::format::nhwc)) {
        input_format = memory::format::x;
      }
      if (src_tz.size() == 2 && (input_format == memory::format::nchw ||
                                 input_format == memory::format::nhwc)) {
        input_format = memory::format::nc;
      }

      for (int i = in_place ? 1 : 0; i < N; i++) {
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
    } else if (out_var->IsType<framework::SelectedRows>()) {
      // TODO(@mozga-intel) Add MKLDNN SelectedRows support
      std::unique_ptr<framework::SelectedRows> in0;
      if (in_place) {
        // If is in_place, we store the input[0] to in0
        auto& in_sel0 = in_vars[0]->Get<SelectedRows>();
        auto& rows = in_sel0.rows();
        in0.reset(new framework::SelectedRows(rows, in_sel0.height()));
        in0->mutable_value()->ShareDataWith(in_sel0.value());
      }

      auto get_selected_row = [&](size_t i) -> const SelectedRows& {
        if (i == 0 && in0) {
          return *in0.get();
        } else {
          return in_vars[i]->Get<SelectedRows>();
        }
      };
      auto* out = ctx.Output<SelectedRows>("Out");
      out->mutable_rows()->clear();
      auto* out_value = out->mutable_value();

      // Runtime InferShape
      size_t first_dim = 0;
      for (int i = 0; i < N; i++) {
        auto& sel_row = get_selected_row(i);
        first_dim += sel_row.rows().size();
      }
      auto in_dim =
          framework::vectorize(get_selected_row(N - 1).value().dims());
      in_dim[0] = static_cast<int64_t>(first_dim);

      out_value->Resize(framework::make_ddim(in_dim));

      // if all the input sparse vars are empty, no need to
      // merge these vars.
      if (first_dim == 0UL) {
        return;
      }
      out_value->mutable_data<T>(ctx.GetPlace());
      math::SelectedRowsAddTo<CPUDeviceContext, T> functor;
      int64_t offset = 0;
      for (int i = 0; i < N; i++) {
        auto& sel_row = get_selected_row(i);
        if (sel_row.rows().size() == 0) {
          continue;
        }
        PADDLE_ENFORCE_EQ(out->height(), sel_row.height());
        functor(ctx.template device_context<CPUDeviceContext>(), sel_row,
                offset, out);
        offset += sel_row.value().numel();
      }
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      // TODO(@mozga-intel) Add MKLDNN LoDTensorArray support
      auto& out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                       "Only support all inputs are TensorArray");
        auto& in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].numel() != 0) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (out_array[i].numel() == 0) {
              framework::TensorCopy(in_array[i], in_array[i].place(),
                                    ctx.device_context(), &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(*ctx.template device_context<MKLDNNDeviceContext>()
                                 .eigen_device()) = result + in;
            }
          }
        }
      }
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   out_var->Type().name());
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(sum, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::SumMKLDNNOpKernel<float>);
