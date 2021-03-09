/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

/*
Tensor topk_assit_help(const Tensor& self, int64_t dim) {
  const int64_t UB_SIZE = 4096;
  assit.resize(2 * UB_SIZE);
  for (int64_t i = 0; i < UB_SIZE; i++) {
    assit[i] = (Half)(i);
  }
   
  for (int64_t i = 0; i < UB_SIZE; i++) {
    int64_t idx = (Half)i;
    int64_t gap = i - idx;
    assit[i + UBSIZE] = (Half)gap;
  }
  Tensor assitHelp = from_blob(assit.data(), {1, 2 * UB_SIZE}, dtype(ScalerType::Half));
  return CalcuOpUtil::copy_tensor_host_to_device(assitHelp);
}*/


template <typename DeviceContext, typename T>
class TopkNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // read input
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");

    size_t k = static_cast<int>(ctx.Attr<int>("k"));

    output->mutable_data<paddle::platform::float16>(ctx.GetPlace());
    indices->mutable_data<int32_t>(ctx.GetPlace());

    // prepare ascend input
    framework::Tensor assist_seq_tensor; //(paddle::platform::float16);
    assist_seq_tensor.Resize({6});
    assist_seq_tensor.mutable_data<paddle::platform::float16>(ctx.GetPlace());
    std::vector<paddle::platform::float16> assist_seq_vec;

    //assist_seq_vec.push_back(                    static_cast<paddle::platform::float16>(1));
    //assist_seq_vec.push_back(                    static_cast<paddle::platform::float16>(2));
    //assist_seq_vec.push_back(                    static_cast<paddle::platform::float16>(3));
    //assist_seq_vec.push_back(static_cast<int>(1)-static_cast<paddle::platform::float16>(1));
    //assist_seq_vec.push_back(static_cast<int>(2)-static_cast<paddle::platform::float16>(2));
    //assist_seq_vec.push_back(static_cast<int>(3)-static_cast<paddle::platform::float16>(3));
    std::cout << "111111111111" << std::endl;

    assist_seq_vec.push_back(static_cast<paddle::platform::float16>(1));
    //assist_seq_vec.push_back(static_cast<paddle::platform::float16>(2));
    //assist_seq_vec.push_back(static_cast<paddle::platform::float16>(3));
    assist_seq_vec.push_back(static_cast<paddle::platform::float16>(1));
    //assist_seq_vec.push_back(static_cast<paddle::platform::float16>(2));
    //assist_seq_vec.push_back(static_cast<paddle::platform::float16>(3));

    std::cout << "222222222222222" << std::endl;

    framework::TensorFromVector(assist_seq_vec, ctx.device_context(), &assist_seq_tensor);
    framework::NPUAttributeMap attr_input = {{"sorted", false}, {"k", static_cast<int>(k)}, {"dim", -1}, {"largest", true}};
    //framework::NPUAttributeMap attr_input = {{"k", static_cast<int>(k)}};

    std::cout << "333333333333333" << std::endl;

    // run ascend
    //
    auto runner = NpuOpRunner("TopKD",
                              {*input, assist_seq_tensor},
                              {*output, *indices}, 
                              attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    runner.Run(stream);
    /*
    std::cout << "after run "<<std::endl;
    framework::Tensor cpu_tensor;
    framework::TensorCopySync(*indices, platform::CPUPlace(), &cpu_tensor);
    auto* data = cpu_tensor.data<T>();
    auto vec_data = std::vector<T>(data, data + indices->numel());
    for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
       VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
    }
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    top_k,
    ops::TopkNPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>);
#endif

