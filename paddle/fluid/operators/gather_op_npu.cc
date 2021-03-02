/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {

template <typename T>
void PrintTensor(const Tensor& src, const framework::ExecutionContext& ctx){
    std::vector<T> vec(src.numel());
    TensorToVector(src, ctx.device_context(), &vec);
    for(int i=0; i< static_cast<int>(vec.size()); ++i){
        VLOG(3) << "vec[" << i<< "] : "<< vec[i];
    }
}

inline framework::Tensor UnsqueezeTo(const framework::Tensor& src, int ndims) {
  const framework::DDim& shape = src.dims();
  int rank = shape.size();
  framework::Tensor res;
  res.ShareDataWith(src);
  PADDLE_ENFORCE_LE(
      rank, ndims,
      platform::errors::InvalidArgument(
          "The input Tensor's rank should be less than or equal to ndims"
          "Received input Tensor's rank = %d, ndims = %d",
          rank, ndims));
  if (rank < ndims) {
    std::vector<int64_t> new_dim(ndims, 1);
    for (int i = ndims - rank; i < ndims; i++) {
      new_dim[i] = shape[i - ndims + rank];
    }
    res.Resize(framework::make_ddim(new_dim));
  }
  return res;
}

template <typename DeviceContext, typename T>
class GatherOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *out = ctx.Output<Tensor>("Out");
    
    out->mutable_data<T>(ctx.GetPlace());
    auto runner = NpuOpRunner("Gather", {*x, *index}, {*out}, {{"validate_indices", true}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GatherGradOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *index = ctx.Input<Tensor>("Index");
    auto *x = ctx.Input<Tensor>("X");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    
    // step1: Unsqueeze index
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      framework::Tensor tmp_index = UnsqueezeTo(*index, 2);
      index = &tmp_index;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    /**
    Tensor* tmp_index = const_cast<Tensor*>(index);
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      Tensor unsqueeze_out(index->type());
      VLOG(3) << "-------index_dims[0]---------" << index_dims[0];
      unsqueeze_out.Resize(paddle::framework::make_ddim({1, index_dims[0]}));
      unsqueeze_out.mutable_data<T>(ctx.GetPlace());
      
      std::vector<int> axes = {1};
      auto runner_unsqueeze = NpuOpRunner("Unsqueeze", {*index}, {unsqueeze_out}, {{"axes", axes}});
      runner_unsqueeze.Run(stream);
      tmp_index = &unsqueeze_out;
    }*/ 
 
    // step2: ZerosLike x in device 
    Tensor* tmp_zerox = const_cast<Tensor*>(x);
    Tensor zeroslike_xout(x->type());
    zeroslike_xout.Resize(x->dims());
    zeroslike_xout.mutable_data<T>(ctx.GetPlace());

    auto runner_zeroslike = NpuOpRunner("ZerosLike", {*x}, {zeroslike_xout}, {});
    runner_zeroslike.Run(stream);
    tmp_zerox = &zeroslike_xout;
    
    ctx.device_context().Wait();

    // step3: scatter(x_grad)
    dx->mutable_data<T>(ctx.GetPlace());
    auto runner_scatter = NpuOpRunner("TensorScatterUpdate", 
                                        {*tmp_zerox, *index, *dout}, 
                                        {*dx}, {});
    runner_scatter.Run(stream);
    // TODO why ascend_parser return [tensor_zeros, x_grad] ?
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    gather, 
    ops::GatherOpNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GatherOpNPUKernel<paddle::platform::NPUDeviceContext, 
                           paddle::platform::float16>);
                           
REGISTER_OP_NPU_KERNEL(
    gather_grad, 
    ops::GatherGradOpNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GatherGradOpNPUKernel<paddle::platform::NPUDeviceContext, 
                               paddle::platform::float16>);
#endif
