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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

//#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/range_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename T>
class NPURangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");

    framework::Tensor n;
    framework::TensorCopySync(*start_t, platform::CPUPlace(), &n);
    T start = n.data<T>()[0];
    framework::TensorCopySync(*end_t, platform::CPUPlace(), &n);
    T end = n.data<T>()[0];
    framework::TensorCopySync(*step_t, platform::CPUPlace(), &n);
    T step = n.data<T>()[0];

    //std::cout << "start>>>>>>>>>>>>>>>"<<  start << "----------" <<std::endl;
    //std::cout << "end>>>>>>>>>>>>>>>"  <<  end << "----------" <<std::endl ;
    //std::cout << "step>>>>>>>>>>>>>>>" <<  step << "----------"  <<std::endl;

    int64_t size = 0;
    GetSize(start, end, step, &size);
    out->Resize(framework::make_ddim({size}));
    //std::cout << "dims>>>>>>>>>>>>>>>" <<  out->dims()[0] << "----------"  <<std::endl;
    out->mutable_data<T>(context.GetPlace());

    //std::cout << "start dims>>>>>>>>>>>>>>>" <<  start_t->dims()[0] << "----------"  <<std::endl;
    //((framework::Tensor*) start_t)->Resize(framework::make_ddim({}));
    //((framework::Tensor*) end_t)->Resize(framework::make_ddim({}));
    //((framework::Tensor*) step_t)->Resize(framework::make_ddim({}));
    //((framework::Tensor*) start_t)->Resize({});
    //((framework::Tensor*) end_t)->Resize({});
    //((framework::Tensor*) step_t)->Resize({});

    auto runner = NpuOpRunner("Range", {*start_t, *end_t, *step_t}, {*out}, {});
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(range, 
                       ops::NPURangeKernel<float>,
                       ops::NPURangeKernel<int>,
                       ops::NPURangeKernel<int64_t>,
                       ops::NPURangeKernel<double>);

#endif

/*
framework::Tensor cpu_tensor;
TensorCopySync(*start_t, platform::CPUPlace(), &cpu_tensor);

auto* data = cpu_tensor.data<T>();

auto vec_data = std::vector<T>(data, data + start_t->numel());
for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
}


TensorCopySync(*end_t, platform::CPUPlace(), &cpu_tensor);
data = cpu_tensor.data<T>();
vec_data = std::vector<T>(data, data + start_t->numel());
for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
}

TensorCopySync(*step_t, platform::CPUPlace(), &cpu_tensor);
data = cpu_tensor.data<T>();
vec_data = std::vector<T>(data, data + start_t->numel());
for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
}


TensorCopySync(*out, platform::CPUPlace(), &cpu_tensor);
data = cpu_tensor.data<T>();
vec_data = std::vector<T>(data, data + start_t->numel());
for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << " vec_data["<< i << "] = " << vec_data[i];
}
*/
