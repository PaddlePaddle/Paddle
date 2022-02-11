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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

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

static void CastToFP32(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream, const Tensor& in,
                       Tensor* out) {
  out->mutable_data<float>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT)
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

    Tensor indices_tmp(experimental::DataType::INT32);
    indices_tmp.Resize(indices->dims());

    if (framework::TransToProtoVarType(input->dtype()) ==
        framework::proto::VarType::INT64) {
      Tensor input_fp32(experimental::DataType::FLOAT32);
      input_fp32.Resize(input->dims());
      CastToFP32(ctx, stream, *input, &input_fp32);

      Tensor output_fp32(experimental::DataType::FLOAT32);
      output_fp32.Resize(output->dims());

      if (axis == -1 || axis + 1 == in_dims.size()) {
        output_fp32.mutable_data<float>(ctx.GetPlace());
        indices_tmp.mutable_data<int32_t>(ctx.GetPlace());
        const auto& runner =
            NpuOpRunner("Sort", {input_fp32}, {output_fp32, indices_tmp}, attr);
        runner.Run(stream);

        CastToInt64(ctx, stream, output_fp32, output);
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

        Tensor trans_input(input_fp32.type());
        trans_input.Resize(trans_dims);
        TranposeNPU<float>(ctx, stream, &perm, input_fp32, &trans_input);

        Tensor trans_output(input_fp32.type());
        Tensor trans_indices(experimental::DataType::INT32);
        trans_output.mutable_data<float>(trans_dims, ctx.GetPlace());
        trans_indices.mutable_data<int32_t>(trans_dims, ctx.GetPlace());

        const auto& runner = NpuOpRunner("Sort", {trans_input},
                                         {trans_output, trans_indices}, attr);
        runner.Run(stream);

        TranposeNPU<float>(ctx, stream, &perm, trans_output, &output_fp32);
        TranposeNPU<int32_t>(ctx, stream, &perm, trans_indices, &indices_tmp);

        CastToInt64(ctx, stream, output_fp32, output);
      }
    } else {
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

        std::vector<int64_t> shape;
        for (size_t i = 0; i < perm.size(); i++) {
          shape.emplace_back(in_dims[perm[i]]);
        }
        auto trans_dims = framework::make_ddim(shape);

        Tensor trans_input(input->type());
        trans_input.Resize(trans_dims);
        TranposeNPU<T>(ctx, stream, &perm, *input, &trans_input);

        Tensor trans_output(input->type());
        Tensor trans_indices(experimental::DataType::INT32);
        trans_output.mutable_data<T>(trans_dims, ctx.GetPlace());
        trans_indices.mutable_data<int32_t>(trans_dims, ctx.GetPlace());

        const auto& runner = NpuOpRunner("Sort", {trans_input},
                                         {trans_output, trans_indices}, attr);
        runner.Run(stream);

        TranposeNPU<T>(ctx, stream, &perm, trans_output, output);
        TranposeNPU<int32_t>(ctx, stream, &perm, trans_indices, &indices_tmp);
      }
    }

    CastToInt64(ctx, stream, indices_tmp, indices);
  }
};

template <typename T, typename Type>
static void FullAssignNPU(const framework::ExecutionContext& ctx,
                          const aclrtStream& stream,
                          const framework::DDim in_dims, const Tensor& input,
                          const Tensor& indices, Tensor* t_out) {
  const int64_t input_height =
      framework::product(framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
  const int64_t input_width = in_dims[in_dims.size() - 1];

  Tensor input_tmp;
  input_tmp.ShareDataWith(input);
  input_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height * input_width}));

  Tensor indices_tmp;
  indices_tmp.ShareDataWith(indices);
  indices_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height, input_width}));

  std::vector<int64_t> indexs_value;
  for (Type i = 0; i < input_height; i++) {
    indexs_value.push_back(i * input_width);
  }
  Tensor indexs_tmp(indices.type());
  framework::TensorFromVector<int64_t>(indexs_value, ctx.device_context(),
                                       &indexs_tmp);
  indexs_tmp.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height, 1}));

  Tensor indices_index(indices.type());
  indices_index.mutable_data<int64_t>(indices_tmp.dims(), ctx.GetPlace());
  const auto& runner_add =
      NpuOpRunner("Add", {indices_tmp, indexs_tmp}, {indices_index}, {});
  runner_add.Run(stream);

  indices_index.Resize(
      framework::make_ddim(std::vector<int64_t>{input_height * input_width}));

  t_out->mutable_data<T>(ctx.GetPlace());
  Tensor out_tmp(t_out->type());
  out_tmp.ShareDataWith(*t_out);

  const auto& runner =
      NpuOpRunner("TensorScatterUpdate", {input_tmp, indices_index, input_tmp},
                  {out_tmp}, {});
  runner.Run(stream);
}

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

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    if (axis == -1 || axis + 1 == in_dims.size()) {
      FullAssignNPU<T, int64_t>(ctx, stream, in_dims, *dO, *indices, dX);
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

      TranposeNPU<T>(ctx, stream, &perm, *dO, &trans_dout);
      TranposeNPU<int64_t>(ctx, stream, &perm, *indices, &trans_ids);

      Tensor trans_dx(dO->type());
      trans_dx.Resize(trans_dims);
      FullAssignNPU<T, int64_t>(ctx, stream, trans_dims, trans_dout, trans_ids,
                                &trans_dx);

      TranposeNPU<T>(ctx, stream, &perm, trans_dx, dX);
    }
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
                       ops::ArgsortGradNPUKernel<paddle::platform::float16>);
