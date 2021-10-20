/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/argsort_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
// #include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

// static void // PrintTensor(const Tensor* tensor, const std::string name,
//                         const std::string msg) {
//   std::cout << "=================== Print Tensor <" << name << ">, Place <"
//             << tensor->place() << "> ===================" << std::endl;
//   framework::LoDTensor cpu_tensor;
//   cpu_tensor.Resize(tensor->dims());
//   framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

//   operators::TensorFormatter formatter;
//   formatter.Print(cpu_tensor, name, msg);
// }

template <typename T>
static void TranposeNPU(const framework::ExecutionContext& ctx,
                        const aclrtStream& stream, std::vector<int64_t>* perm,
                        const Tensor& in, Tensor* out) {
  out->mutable_data<T>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(in)
      .AddInput(std::move(*perm))
      .AddOutput(*out)
      .Run(stream);
}

static void CastToInt64(const framework::ExecutionContext& ctx,
                        const aclrtStream& stream, const Tensor& in,
                        Tensor* out) {
  out->mutable_data<int64_t>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_INT64)
      .Run(stream);
}

template <typename T>
class ArgsortNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int axis = ctx.Attr<int>("axis");
    bool descending = ctx.Attr<bool>("descending");

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    framework::NPUAttributeMap attr = {{"axis", -1},
                                       {"descending", descending}};

    Tensor indices_tmp(framework::proto::VarType::INT32);
    indices_tmp.Resize(indices->dims());

    // PrintTensor(input, "input", "forward");
    if (axis == -1 || axis + 1 == in_dims.size()) {
      output->mutable_data<T>(ctx.GetPlace());
      indices_tmp.mutable_data<int32_t>(ctx.GetPlace());
      const auto& runner =
          NpuOpRunner("Sort", {*input}, {*output, indices_tmp}, attr);
      runner.Run(stream);
    } else {
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < in_dims.size(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[axis], perm[in_dims.size() - 1]);

      // LOG(INFO) << "perm = " << framework::make_ddim(perm).to_str();

      std::vector<int64_t> shape;
      for (size_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(in_dims[perm[i]]);
      }
      auto trans_dims = framework::make_ddim(shape);

      // LOG(INFO) << "trans_dims = " << trans_dims.to_str();

      Tensor trans_input(input->type());
      trans_input.Resize(trans_dims);
      TranposeNPU<T>(ctx, stream, &perm, *input, &trans_input);

      // PrintTensor(&trans_input, "trans_input", "forward");

      Tensor trans_output(input->type());
      Tensor trans_indices(framework::proto::VarType::INT32);
      trans_output.mutable_data<T>(trans_dims, ctx.GetPlace());
      trans_indices.mutable_data<int32_t>(trans_dims, ctx.GetPlace());

      const auto& runner = NpuOpRunner("Sort", {trans_input},
                                       {trans_output, trans_indices}, attr);
      runner.Run(stream);

      // PrintTensor(&trans_output, "trans_output", "forward");
      // PrintTensor(&trans_indices, "trans_indices", "forward");

      TranposeNPU<T>(ctx, stream, &perm, trans_output, output);
      TranposeNPU<int32_t>(ctx, stream, &perm, trans_indices, &indices_tmp);
    }
    // PrintTensor(&indices_tmp, "indices_tmp", "forward");
    CastToInt64(ctx, stream, indices_tmp, indices);
    // PrintTensor(output, "output", "forward");
    // PrintTensor(indices, "indices", "forward");
  }
};

// template <typename Type>
// static void ReshapeNPU(const framework::Tensor* input,
//                        const std::vector<Type>& input_shapes,
//                        framework::Tensor* output) {
//   output->ShareDataWith(*input);
//   output->Resize(framework::make_ddim(std::move(input_shapes)));
// }

template <typename T, typename Type>
static void FullAssignNPU(const framework::ExecutionContext& ctx,
                          const aclrtStream& stream,
                          const framework::DDim in_dims, const Tensor* input,
                          const Tensor* indices, Tensor* t_out) {
  // // LOG(INFO) << "in_dims = " << in_dims.to_str();

  const int64_t input_height =
      framework::product(framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
  const int64_t input_width = in_dims[in_dims.size() - 1];

  // // LOG(INFO) << "input_height = " << input_height;
  // // LOG(INFO) << "input_width = " << input_width;

  Tensor input_tmp;
  input_tmp.ShareDataWith(*input);
  input_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height * input_width}));

  // // PrintTensor(&input_tmp, "input_tmp", "backward");

  Tensor indices_tmp;
  indices_tmp.ShareDataWith(*indices);
  indices_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height, input_width}));

  // // PrintTensor(&indices_tmp, "indices_tmp", "backward");

  std::vector<int64_t> indexs_value;
  for (Type i = 0; i < input_height; i++) {
    indexs_value.push_back(i * input_width);
  }
  Tensor indexs_tmp(indices->type());
  framework::TensorFromVector<int64_t>(indexs_value, ctx.device_context(),
                                       &indexs_tmp);
  indexs_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height, 1}));

  // // PrintTensor(&indexs_tmp, "indexs_tmp", "backward");

  Tensor indices_index(indices->type());
  indices_index.mutable_data<int64_t>(indices_tmp.dims(), ctx.GetPlace());
  const auto& runner_add =
      NpuOpRunner("Add", {indices_tmp, indexs_tmp}, {indices_index}, {});
  runner_add.Run(stream);
  // // PrintTensor(&indices_index, "indices_index", "backward");

  indices_index.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height * input_width}));

  // // PrintTensor(&indices_index, "indices_index", "backward");

  t_out->mutable_data<T>(ctx.GetPlace());
  Tensor out_tmp(t_out->type());
  out_tmp.ShareDataWith(*t_out);

  const auto& runner =
      NpuOpRunner("TensorScatterUpdate", {input_tmp, indices_index, input_tmp},
                  {out_tmp}, {});
  runner.Run(stream);

  // // PrintTensor(&out_tmp, "out_tmp", "backward");
}

// template <typename T, typename Type>
// static void FullAssignNPU(const framework::ExecutionContext& ctx,
//                           Type ind_lastdim, Type outer_dim,
//                           const framework::DDim& trans_dims,
//                           const framework::Tensor* input,
//                           const framework::Tensor* indices,
//                           framework::Tensor* t_out) {
//   // // reshape input
//   // Type input_shape = ind_lastdim * outer_dim;
//   // std::vector<Type> input_shapes = {input_shape};
//   // Tensor input_reshape_tensor(input->type());
//   // ReshapeNPU<Type>(input, input_shapes, &input_reshape_tensor);
//   // // reshape index
//   // std::vector<Type> index_shapes = {outer_dim, ind_lastdim};
//   // framework::DDim ind_2d = framework::make_ddim({outer_dim, ind_lastdim});
//   // Tensor ind_2d_tensor(indices->type());
//   // ReshapeNPU<Type>(indices, index_shapes, &ind_2d_tensor);
//   // range_flatten_index
//   std::vector<int32_t> range_flatten_index;
//   for (Type i = 0; i < input_shape; i += ind_lastdim) {
//     range_flatten_index.push_back(static_cast<int32_t>(i));
//   }
//   Tensor range_flatten_index_tensor(framework::proto::VarType::INT32);
//   range_flatten_index_tensor.Resize(framework::make_ddim({outer_dim}));
//   range_flatten_index_tensor.mutable_data<int32_t>(
//       {static_cast<int>(range_flatten_index.size())}, ctx.GetPlace());
//   TensorFromVector(range_flatten_index, ctx.device_context(),
//                    &range_flatten_index_tensor);
//   Tensor
//   range_flatten_index_expand_tensor(range_flatten_index_tensor.type());
//   std::vector<Type> flatten_shape = {outer_dim, 1};
//   ReshapeNPU<Type>(&range_flatten_index_tensor, flatten_shape,
//                    &range_flatten_index_expand_tensor);

//   // range_flatten_index_expand_tensor = (6 x 1) = {0, 4, 8, 12, 16, 20}
//   Tensor ind_2d_add_tensor;
//   ind_2d_add_tensor.mutable_data<int32_t>(ind_2d, ctx.GetPlace());
//   const auto& runner_ind_2d_tensor = NpuOpRunner(
//       std::string("Add"), {ind_2d_tensor, range_flatten_index_expand_tensor},
//       {ind_2d_add_tensor}, {});
//   runner_ind_2d_tensor.Run(stream);
//   // = index + {0, 4, 8, 12, 16, 20} = [0, 1, 2, 3] [4, 5, 6, 7] ...

//   Tensor ind_reshape_tensor(ind_2d_add_tensor.type());
//   ReshapeNPU<Type>(&ind_2d_add_tensor, input_shapes, &ind_reshape_tensor);
//   // ind_2d_add_tensor = (24)
//   Tensor ind_reshape_expand_tensor(ind_reshape_tensor.type());
//   std::vector<Type> ind_shape = {input_shape, 1};
//   ReshapeNPU<Type>(&ind_reshape_tensor, ind_shape,
//   &ind_reshape_expand_tensor);
//   // ind_reshape_expand_tensor = (24 x 1)

//   // expand_index
//   Tensor input_scatter_tensor;
//   input_scatter_tensor.Resize({input_shape});
//   input_scatter_tensor.mutable_data<T>(ctx.GetPlace());
//   Tensor input_scatter_tensor_ori;
//   input_scatter_tensor_ori.Resize({input_shape});
//   input_scatter_tensor_ori.mutable_data<T>(ctx.GetPlace());
//   std::vector<Type> trans_shapes;

//   for (int i = 0; i < trans_dims.size(); i++) {
//     trans_shapes.push_back(trans_dims[i]);
//   }
//   NpuOpRunner runner_scatter;
//   runner_scatter.SetType("TensorScatterUpdate")
//       .AddInput(input_scatter_tensor_ori) // None
//       .AddInput(ind_reshape_expand_tensor) // (24 x 1) =[1, 1, 1, 1] [5, 5,
//       5, 5] ...
//       .AddInput(input_reshape_tensor) // (24) = [1, 1, 1, 1] [1, 1, 1, 1] ...
//       .AddOutput(input_scatter_tensor);
//   runner_scatter.Run(stream);
//   framework::TensorCopy(input_scatter_tensor, ctx.GetPlace(),
//                         ctx.template
//                         device_context<platform::DeviceContext>(),
//                         t_out);
//   t_out->Resize(framework::make_ddim(trans_shapes));
// }

template <typename T>
class ArgsortGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int axis = ctx.Attr<int>("axis");

    auto in_dims = indices->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    if (dO->numel() == 0) return;

    // dX->mutable_data<T>(ctx.GetPlace());

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // // PrintTensor(dO, "dO", "backward");

    // Do full assign
    if (axis == -1 || axis + 1 == in_dims.size()) {
      // const int64_t input_height = framework::product(
      //     framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      // const int64_t input_width = in_dims[in_dims.size() - 1];

      // // LOG(INFO) << "input_height = " << input_height;
      // // LOG(INFO) << "input_width = " << input_width;

      FullAssignNPU<T, int64_t>(ctx, stream, in_dims, dO, indices, dX);
    } else {
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < in_dims.size(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[axis], perm[in_dims.size() - 1]);

      std::vector<int64_t> shape;
      for (size_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(in_dims[perm[i]]);
      }
      auto trans_dims = framework::make_ddim(shape);

      Tensor trans_dout(dO->type());
      Tensor trans_ids(indices->type());
      trans_dout.Resize(trans_dims);
      trans_ids.Resize(trans_dims);
      // trans_dout.mutable_data<T>(trans_dims, ctx.GetPlace());
      // trans_ids.mutable_data<int64_t>(trans_dims, ctx.GetPlace());

      TranposeNPU<T>(ctx, stream, &perm, *dO, &trans_dout);
      TranposeNPU<int64_t>(ctx, stream, &perm, *indices, &trans_ids);

      // // PrintTensor(&trans_dout, "trans_dout", "backward");
      // // PrintTensor(&trans_ids, "trans_ids", "backward");

      // const int64_t input_height = framework::product(
      //     framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      // const int64_t input_width = trans_dims[trans_dims.size() - 1];

      // // LOG(INFO) << "input_height = " << input_height;
      // // LOG(INFO) << "input_width = " << input_width;

      Tensor trans_dx(dO->type());
      trans_dx.Resize(trans_dims);
      FullAssignNPU<T, int64_t>(ctx, stream, trans_dims, &trans_dout,
                                &trans_ids, &trans_dx);

      // // PrintTensor(&trans_dx, "trans_dx", "backward");

      TranposeNPU<T>(ctx, stream, &perm, trans_dx, dX);

      // // PrintTensor(dX, "dX", "backward");
    }

    // // Do full assig  n
    // if (axis == -1 || axis + 1 == in_dims.size()) {
    //   const int64_t outer_dim = framework::product(
    //       framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
    //   const int64_t ind_lastdim = in_dims[in_dims.size() - 1];
    //   FullAssignNPU<T, int64_t>(ctx, ind_lastdim, outer_dim, in_dims, dO,
    //   indices, dX);

    // } else {
    //   // If not full assign do transpose
    //   std::vector<int> trans;
    //   for (int i = 0; i < axis; i++) {
    //     trans.push_back(i);
    //   }
    //   trans.push_back(in_dims.size() - 1);
    //   for (int i = axis + 1; i < in_dims.size() - 1; i++) {
    //     trans.push_back(i);
    //   }
    //   trans.push_back(axis);
    //   framework::DDim trans_dims(in_dims);
    //   for (size_t i = 0; i < trans.size(); i++) {
    //     trans_dims[i] = in_dims[trans[i]];
    //   }
    //   std::vector<int> axis;
    //   for (size_t i = 0; i < trans.size(); i++) {
    //     axis.push_back(in_dims[trans[i]]);
    //   }
    //   framework::NPUAttributeMap attr_input = {{"perm", trans}};
    //   Tensor trans_dO;
    //   trans_dO.mutable_data<T>(trans_dims, ctx.GetPlace());
    //   Tensor trans_ind;
    //   trans_ind.mutable_data<int32_t>(trans_dims, ctx.GetPlace());
    //   // Do transpose
    //   const auto& runner_transpose_dx = NpuOpRunner(
    //       std::string("TransposeD"), {*dO}, {trans_dO}, {attr_input});
    //   runner_transpose_dx.Run(stream);
    //   const auto& runner_transpose_ind = NpuOpRunner(
    //       std::string("TransposeD"), {*indices}, {trans_ind}, {attr_input});
    //   runner_transpose_ind.Run(stream);

    //   const int64_t outer_dim = framework::product(
    //       framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    //   const int64_t ind_lastdim = trans_dims[trans_dims.size() - 1];

    //   Tensor tmp_out;
    //   tmp_out.mutable_data<T>(trans_dims, ctx.GetPlace());

    //   FullAssignNPU<T, int64_t>(ctx, ind_lastdim, outer_dim, trans_dims,
    //                             &trans_dO, &trans_ind, &tmp_out);

    //   // transpose back
    //   const auto& runner_transpose_out = NpuOpRunner(
    //       std::string("TransposeD"), {tmp_out}, {*dX}, {attr_input});
    //   runner_transpose_out.Run(stream);
    // }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(argsort, ops::ArgsortNPUKernel<float>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::ArgsortNPUKernel<int64_t>,
#endif
                       ops::ArgsortNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(argsort_grad, ops::ArgsortGradNPUKernel<float>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::ArgsortGradNPUKernel<int64_t>,
#endif
                       ops::ArgsortGradNPUKernel<paddle::platform::float16>);
