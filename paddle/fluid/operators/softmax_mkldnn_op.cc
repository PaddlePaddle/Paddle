/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "mkldnn.hpp"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::softmax_forward;
using mkldnn::prop_kind;
using mkldnn::stream;

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE(input->dims().size() == 2UL,
                   "The input of softmax op must be a 2D matrix.");
    const T* input_data = input->data<T>();
    // allocate memory for output
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    // MKL-DNN does support softmax over selected axis. Having 2D Tensor,
    // we will make normalization after final eg. axis: 1
    PADDLE_ENFORCE(((src_tz[0] == dst_tz[0]) && (src_tz[1] == dst_tz[1])),
                   "Softmax input and output dimensions should match");
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Generate keys for storing/retriving primitives for this operator
    // TODO(jczaja): Each MKLDNN operator may have diffrent hashing function
    auto gethash = [](memory::dims& operand_dims) {
      return std::string(std::to_string(operand_dims[0]) + "-" +
                         std::to_string(operand_dims[1]));
    };
    const std::string key = gethash(softmax_tz);
    const std::string key_softmax_p = key + "@softmax_p";
    const std::string key_softmax_src_mem_p = key + "@softmax_src_mem_p";
    const std::string key_softmax_dst_mem_p = key + "@softmax_dst_mem_p";

    std::shared_ptr<void> softmax_p = dev_ctx.GetBlob(key_softmax_p);
    if (softmax_p == nullptr) {
      // Currently only NC data format is supported
      auto softmax_md =
          MKLDNNMemDesc({softmax_tz}, memory::f32, memory::format::nc);
      // Normalization is made after innermost dimension eg. C out of NC
      auto softmax_desc = softmax_forward::desc(prop_kind::forward_scoring,
                                                softmax_md, 1 /*dim: C*/);
      // create memory primitives
      auto softmax_src_memory_p = std::make_shared<memory>(
          memory::primitive_desc{softmax_md, mkldnn_engine},
          static_cast<void*>(const_cast<T*>(input_data)));
      dev_ctx.SetBlob(key_softmax_src_mem_p, softmax_src_memory_p);
      auto softmax_dst_memory_p = std::make_shared<memory>(
          memory::primitive_desc{softmax_md, mkldnn_engine},
          static_cast<void*>(output_data));
      dev_ctx.SetBlob(key_softmax_dst_mem_p, softmax_dst_memory_p);

      auto softmax_forward_pd =
          std::make_shared<softmax_forward::primitive_desc>(softmax_desc,
                                                            mkldnn_engine);
      softmax_p = std::make_shared<softmax_forward>(
          *(softmax_forward_pd.get()),
          *(static_cast<memory*>(softmax_src_memory_p.get())),
          *(static_cast<memory*>(softmax_dst_memory_p.get())));
      dev_ctx.SetBlob(key_softmax_p, softmax_p);
    } else {
      // Primitives already exist
      auto src_memory_p = std::static_pointer_cast<memory>(
          dev_ctx.GetBlob(key_softmax_src_mem_p));
      PADDLE_ENFORCE(src_memory_p != nullptr,
                     "Fail to find softmax src mem_p in device context");
      auto dst_memory_p = std::static_pointer_cast<memory>(
          dev_ctx.GetBlob(key_softmax_dst_mem_p));
      PADDLE_ENFORCE(dst_memory_p != nullptr,
                     "Fail to find softmax dst mem_p in device context");
      src_memory_p->set_data_handle(
          reinterpret_cast<void*>(const_cast<T*>(input_data)));
      dst_memory_p->set_data_handle(output_data);
    }

    std::vector<primitive> pipeline{
        *(static_cast<softmax_forward::primitive*>(softmax_p.get()))};
    stream(stream::kind::eager).submit(pipeline).wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T threshold = exp(-64);
      for (int i = 0; i < dst_tz[0] * dst_tz[1]; ++i) {
        output_data[i] =
            output_data[i] < threshold ? threshold : output_data[i];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
