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

#include "paddle/fluid/operators/argsort_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ArgsortNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    indices->mutable_data<int32_t>(ctx.GetPlace());

    int32_t axis = ctx.Attr<int>("axis");
    auto in_dims = indices->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    bool descending = ctx.Attr<bool>("descending");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    framework::NPUAttributeMap sort_attr_input = {
        {"axis", static_cast<int32_t>(-1)}, {"descending", descending}};

    if (axis == -1 || axis + 1 == in_dims.size()) {
      const auto& sort_runner =
          NpuOpRunner("Sort", {*input}, {*output, *indices}, sort_attr_input);
      sort_runner.Run(stream);
    } else {
      // transpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.push_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.push_back(i);
      }
      trans.push_back(axis);
      framework::DDim trans_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }
      framework::NPUAttributeMap trans_attr_input = {{"perm", trans}};
      Tensor trans_input;
      trans_input.mutable_data<T>(trans_dims, ctx.GetPlace());
      const auto& trans_input_runner =
          NpuOpRunner("TransposeD", {*input}, {trans_input}, trans_attr_input);
      trans_input_runner.Run(stream);
      Tensor trans_indices;
      trans_indices.mutable_data<int32_t>(trans_dims, ctx.GetPlace());
      const auto& trans_indice_runner = NpuOpRunner(
          "TransposeD", {*indices}, {trans_indices}, trans_attr_input);
      trans_indice_runner.Run(stream);
      Tensor trans_output;
      trans_output.mutable_data<T>(trans_dims, ctx.GetPlace());
      const auto& trans_output_runner = NpuOpRunner(
          "TransposeD", {*output}, {trans_output}, trans_attr_input);
      trans_output_runner.Run(stream);
      const auto& sort_runner =
          NpuOpRunner("Sort", {trans_input}, {trans_output, trans_indices},
                      sort_attr_input);
      sort_runner.Run(stream);
      // transpose back
      const auto& trans_indices_back_runner = NpuOpRunner(
          "TransposeD", {trans_indices}, {*indices}, trans_attr_input);
      trans_indices_back_runner.Run(stream);
      const auto& trans_output_back_runner = NpuOpRunner(
          "TransposeD", {trans_output}, {*output}, trans_attr_input);
      trans_output_back_runner.Run(stream);
    }
  }
};

template <typename Type>
static void ReshapeNPU(const framework::Tensor* input,
                       const std::vector<Type>& input_shapes,
                       framework::Tensor* output) {
  output->ShareDataWith(*input);
  output->Resize(framework::make_ddim(std::move(input_shapes)));
}

template <typename T, typename Type>
static void FullAssignNPU(const framework::ExecutionContext& ctx,
                          Type ind_lastdim, Type outer_dim,
                          const framework::DDim& trans_dims,
                          const framework::Tensor* input,
                          const framework::Tensor* indices,
                          framework::Tensor* t_out) {
  // reshape input
  Type input_shape = ind_lastdim * outer_dim;
  std::vector<Type> input_shapes = {input_shape};
  Tensor input_reshape_tensor(input->type());
  ReshapeNPU<Type>(input, input_shapes, &input_reshape_tensor);
  // reshape index
  std::vector<Type> index_shapes = {outer_dim, ind_lastdim};
  framework::DDim ind_2d = framework::make_ddim({outer_dim, ind_lastdim});
  Tensor ind_2d_tensor(indices->type());
  ReshapeNPU<Type>(indices, index_shapes, &ind_2d_tensor);
  // range_flatten_index
  std::vector<int32_t> range_flatten_index;
  for (Type i = 0; i < input_shape; i += ind_lastdim) {
    range_flatten_index.push_back(static_cast<int32_t>(i));
  }
  Tensor range_flatten_index_tensor(framework::proto::VarType::INT32);
  range_flatten_index_tensor.Resize(framework::make_ddim({outer_dim}));
  range_flatten_index_tensor.mutable_data<int32_t>(
      {static_cast<int>(range_flatten_index.size())}, ctx.GetPlace());
  TensorFromVector(range_flatten_index, ctx.device_context(),
                   &range_flatten_index_tensor);
  Tensor range_flatten_index_expand_tensor(range_flatten_index_tensor.type());
  std::vector<Type> flatten_shape = {outer_dim, 1};
  ReshapeNPU<Type>(&range_flatten_index_tensor, flatten_shape,
                   &range_flatten_index_expand_tensor);
  auto stream =
      ctx.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();
  Tensor ind_2d_add_tensor;
  ind_2d_add_tensor.mutable_data<int32_t>(ind_2d, ctx.GetPlace());
  const auto& runner_ind_2d_tensor = NpuOpRunner(
      std::string("Add"), {ind_2d_tensor, range_flatten_index_expand_tensor},
      {ind_2d_add_tensor}, {});
  runner_ind_2d_tensor.Run(stream);
  Tensor ind_reshape_tensor(ind_2d_add_tensor.type());
  ReshapeNPU<Type>(&ind_2d_add_tensor, input_shapes, &ind_reshape_tensor);
  Tensor ind_reshape_expand_tensor(ind_reshape_tensor.type());
  std::vector<Type> ind_shape = {input_shape, 1};
  ReshapeNPU<Type>(&ind_reshape_tensor, ind_shape, &ind_reshape_expand_tensor);
  // expand_index
  Tensor input_scatter_tensor;
  input_scatter_tensor.Resize({input_shape});
  input_scatter_tensor.mutable_data<T>(ctx.GetPlace());
  Tensor input_scatter_tensor_ori;
  input_scatter_tensor_ori.Resize({input_shape});
  input_scatter_tensor_ori.mutable_data<T>(ctx.GetPlace());
  std::vector<Type> trans_shapes;

  for (int i = 0; i < trans_dims.size(); i++) {
    trans_shapes.push_back(trans_dims[i]);
  }
  NpuOpRunner runner_scatter;
  runner_scatter.SetType("TensorScatterUpdate")
      .AddInput(input_scatter_tensor_ori)
      .AddInput(ind_reshape_expand_tensor)
      .AddInput(input_reshape_tensor)
      .AddOutput(input_scatter_tensor);
  runner_scatter.Run(stream);
  framework::TensorCopy(input_scatter_tensor, ctx.GetPlace(),
                        ctx.template device_context<platform::DeviceContext>(),
                        t_out);
  t_out->Resize(framework::make_ddim(trans_shapes));
}

template <typename DeviceContext, typename T>
class ArgsortGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int axis = ctx.Attr<int>("axis");
    auto in_dims = indices->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    dX->mutable_data<T>(ctx.GetPlace());
    Tensor dxt;
    dxt.mutable_data<T>(dX->dims(), place);
    const auto& runner_flatten =
        NpuOpRunner(std::string("Flatten"), {*dX}, {dxt}, {});
    runner_flatten.Run(stream);
    FillNpuTensorWithConstant<T>(&dxt, static_cast<T>(0));
    if (dO->numel() == 0) return;
    // Do full assig  n
    if (axis == -1 || axis + 1 == in_dims.size()) {
      const int64_t outer_dim = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t ind_lastdim = in_dims[in_dims.size() - 1];
      FullAssignNPU<T, int64_t>(ctx, ind_lastdim, outer_dim, in_dims, dO,
                                indices, dX);

    } else {
      // If not full assign do transpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.push_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.push_back(i);
      }
      trans.push_back(axis);
      framework::DDim trans_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }
      std::vector<int> axis;
      for (size_t i = 0; i < trans.size(); i++) {
        axis.push_back(in_dims[trans[i]]);
      }
      framework::NPUAttributeMap attr_input = {{"perm", trans}};
      Tensor trans_dO;
      trans_dO.mutable_data<T>(trans_dims, ctx.GetPlace());
      Tensor trans_ind;
      trans_ind.mutable_data<int32_t>(trans_dims, ctx.GetPlace());
      // Do transpose
      const auto& runner_transpose_dx = NpuOpRunner(
          std::string("TransposeD"), {*dO}, {trans_dO}, {attr_input});
      runner_transpose_dx.Run(stream);
      const auto& runner_transpose_ind = NpuOpRunner(
          std::string("TransposeD"), {*indices}, {trans_ind}, {attr_input});
      runner_transpose_ind.Run(stream);

      const int64_t outer_dim = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t ind_lastdim = trans_dims[trans_dims.size() - 1];

      Tensor tmp_out;
      tmp_out.mutable_data<T>(trans_dims, ctx.GetPlace());

      FullAssignNPU<T, int64_t>(ctx, ind_lastdim, outer_dim, trans_dims,
                                &trans_dO, &trans_ind, &tmp_out);

      // transpose back
      const auto& runner_transpose_out = NpuOpRunner(
          std::string("TransposeD"), {tmp_out}, {*dX}, {attr_input});
      runner_transpose_out.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    argsort, ops::ArgsortNPUKernel<plat::NPUDeviceContext, float>,
    ops::ArgsortNPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(argsort_grad,
                       ops::ArgsortGradNPUKernel<plat::NPUDeviceContext, float>,
                       ops::ArgsortGradNPUKernel<plat::NPUDeviceContext,
                                                 paddle::platform::float16>);
