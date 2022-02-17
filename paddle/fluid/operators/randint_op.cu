// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
class GPURandintKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        context.MultiInput<framework::Tensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || context.HasInput("ShapeTensor")) {
      if (context.HasInput("ShapeTensor")) {
        auto* shape_tensor = context.Input<framework::Tensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    platform::CPUPlace cpu;
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    auto* out = context.Output<framework::LoDTensor>("Out");
    if (!new_shape.empty()) out->Resize(framework::make_ddim(new_shape));
    T low = static_cast<T>(context.Attr<int>("low"));
    T high = static_cast<T>(context.Attr<int>("high")) - 1;
    framework::LoDTensor tensor;
    tensor.Resize(out->dims());
    tensor.mutable_data(cpu, framework::TransToPtenDataType(dtype));
    T* data = tensor.mutable_data<T>(cpu);

    int64_t size = out->numel();
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));

    /*
    std::minstd_rand engine;
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    engine.seed(seed);
    */

    std::uniform_int_distribution<> dist(context.Attr<int>("low"),
                                         context.Attr<int>("high") - 1);
    auto engine = framework::GetCPURandomEngine(seed);

    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(*engine);
    }

    if (platform::is_gpu_place(context.GetPlace())) {
      // Copy tensor to out
      framework::TensorCopy(tensor, context.GetPlace(), out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(randint, ops::GPURandintKernel<int>,
                        ops::GPURandintKernel<int64_t>)
