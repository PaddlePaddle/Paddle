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
        "ConvTransposeMKLDNN works only for inference!. Set is_test = True");
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

    std::vector<int> nchw_axis(ndims, 0);
    for (size_t i = 0; i < nchw_axis.size(); ++i) {
      nchw_axis[i] = i;
    }

    std::vector<int> nchw_tz = paddle::framework::vectorize2int(input->dims());
    std::string data_format = ctx.Attr<std::string>("data_format");

    auto src_md =
        input->format() != mkldnn::memory::format::nchw
            ? platform::MKLDNNMemDesc(nchw_tz, platform::MKLDNNGetDataType<T>(),
                                      input->format())
            : Axis2MemoryDesc(nchw_tz, nchw_axis);

    this->TransposeKernel(ctx.GetPlace(), Axis2MemoryDesc(nchw_tz, axis),
                          src_md, output, input_data, nchw_tz, mkldnn_engine);
  }

 protected:
  mkldnn::memory::desc Axis2MemoryDesc(std::vector<int>& nchw_tz,
                                       std::vector<int>& axis) const {
    mkldnn_memory_desc_t mem_fmt;

    mem_fmt.primitive_kind = mkldnn_memory;
    mem_fmt.ndims = axis.size();
    for (unsigned int i = 0; i < nchw_tz.size(); ++i) {
      mem_fmt.dims[i] = nchw_tz[i];  // logical dimensions (nchw format,
                                     // regardless physical layout)
    }
    mem_fmt.data_type = mkldnn_f32;
    mem_fmt.format = mkldnn_blocked;

    unsigned int total_stride = 1;
    for (int i = nchw_tz.size() - 1; i >= 0; --i) {
      mem_fmt.layout_desc.blocking.padding_dims[i] =
          nchw_tz[i];  // logical dimensions (nchw format, regardless physical
                       // layout)
      mem_fmt.layout_desc.blocking.block_dims[i] = 1;
      mem_fmt.layout_desc.blocking.offset_padding_to_data[i] = 0;  // no offset
      mem_fmt.layout_desc.blocking.strides[0][axis[i]] = total_stride;
      mem_fmt.layout_desc.blocking.strides[1][axis[i]] = 1;
      total_stride *= nchw_tz[axis[i]];
    }
    mem_fmt.layout_desc.blocking.offset_padding = 0;  // no initial offset
    return mem_fmt;
  }

  void TransposeKernel(platform::Place place, mkldnn::memory::desc md_o,
                       mkldnn::memory::desc md_i, Tensor* output,
                       const T* data_i, std::vector<int>& nchw_dims,
                       const mkldnn::engine& eng) const {
    // Make Memory primitive descriptors
    auto mpd_o = mkldnn::memory::primitive_desc(md_o, eng);
    auto mpd_i = mkldnn::memory::primitive_desc(md_i, eng);

    auto data_o = output->mutable_data<T>(
        place, paddle::memory::Allocator::kDefault, mpd_o.get_size());

    auto src = mkldnn::memory(mpd_i, (T*)(data_i));
    auto dst = mkldnn::memory(mpd_o, data_o);

    auto r = mkldnn::reorder(src, dst);
    mkldnn::stream(mkldnn::stream::kind::eager).submit({r}).wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(transpose2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);
