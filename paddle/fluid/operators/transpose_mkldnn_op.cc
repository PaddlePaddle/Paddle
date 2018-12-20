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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using framework::DataLayout;

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    const bool is_test = ctx.Attr<bool>("is_test");
    PADDLE_ENFORCE(
        is_test == true,
        "TransposeMKLDNN works only for inference!. Set is_test = True");
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    const T* input_data = input->data<T>();

    if (ndims == 1) {
      output->ShareDataWith(*input);
      return;
    }

    std::vector<int> nchw_tz = paddle::framework::vectorize2int(input->dims());

    const std::string key = platform::TransposeMKLDNNHandler::GetHash(
        nchw_tz, axis, ctx.op().Output("Out"));

    platform::TransposeMKLDNNHandler handler(nchw_tz, axis, dev_ctx,
                                             mkldnn_engine, key);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        input->format(), platform::to_void_cast<T>(input_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(output, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*transpose_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(transpose2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);
