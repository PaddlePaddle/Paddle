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

#include "paddle/fluid/operators/deformable_conv_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

void PrintTensor(const Tensor* tensor, const std::string& name, const std::string& msg) {
  std::cout << "=================== Print Tensor <" << name << ">, Place <" << tensor->place() << "> ===================" <<std::endl;
  framework::LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

  operators::TensorFormatter formatter;
  formatter.Print(cpu_tensor, name, msg);
}

template <typename T>
void PrintVector(const std::vector<T>& vec, const std::string& name) {
  std::cout << name << " = [";
  for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i] << ", ";
  }
  std::cout << "]" << std::endl;
}

template <typename T>
class DeformableConvNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* offset = ctx.Input<Tensor>("Offset");
    const Tensor* mask = ctx.Input<Tensor>("Mask");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    Tensor* offset_out = ctx.Output<Tensor>("OffsetOut");
    output->mutable_data<T>(ctx.GetPlace());
    offset_out->mutable_data<T>(ctx.GetPlace());

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    // const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::vector<int64_t> kernel_size{filter->dims()[2], filter->dims()[3]};
    const std::string data_layout = framework::DataLayoutToString(input->layout());

    PrintVector<int>(strides, "strides");
    PrintVector<int>(paddings, "paddings");
    PrintVector<int>(dilations, "dilations");
    PrintVector<int64_t>(kernel_size, "kernel_size");

    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    // Tensor output_temp1(output->type());
    // output_temp1.mutable_data<T>(output->dims(), ctx.GetPlace());

    
    

    PrintTensor(input, "input", "Input - forward");
    PrintTensor(filter, "filter", "Input - forward");
    PrintTensor(offset, "offset", "Input - forward");
    PrintTensor(mask, "mask", "Input - forward");

    // concat offset and mask
    // std::vector<Tensor> offset_list;
    // offset_list.push_back(*offset);
    // offset_list.push_back(*mask);
    // const int64_t offset_size = 3 * kernel_size[0] * kernel_size[1], offset->dims()[2];
    const std::vector<int64_t> offset_concat_size{offset->dims()[0], 3 * kernel_size[0] * kernel_size[1], offset->dims()[2], offset->dims()[3]};
    auto offset_concat = ctx.AllocateTmpTensor<T, NPUDeviceContext>(framework::make_ddim(offset_concat_size), dev_ctx);
    NpuOpRunner runner_concat;
    runner_concat.SetType("ConcatD")
        // .AddInput(offset_list)
        .AddInput(*offset)
        .AddInput(*mask)
        .AddOutput(offset_concat)
        .AddAttr("concat_dim", 1)
        .AddAttr("N", 2)
        .AddInputNames({"x0", "x1"})
        .Run(stream);

    PrintTensor(&offset_concat, "offset_concat", "Temp - forward");

    // // DeformableOffsets
    // const int64_t offset_output_h = offset->dims()[2] * kernel_size[0];
    // const int64_t offset_output_w = offset->dims()[3] * kernel_size[1];
    // const std::vector<int64_t> offset_output_size{input->dims()[0], input->dims()[1], offset_output_h, offset_output_w};
    // auto offset_output = ctx.AllocateTmpTensor<T, NPUDeviceContext>(framework::make_ddim(offset_output_size), dev_ctx);

    // LOG(INFO) << "offset_input shape = " << offset_input.dims().to_str();

    const std::vector<int64_t> strides_offset = {1, 1, strides[0], strides[1]};
    const std::vector<int64_t> pads_offset{paddings[0], paddings[0], paddings[1], paddings[1]};
    const std::vector<int64_t> dilations_offset = {1, 1, dilations[0], dilations[1]};

    NpuOpRunner runner_offset;
    runner_offset.SetType("DeformableOffsets")
        .AddInput(*input)
        .AddInput(offset_concat)
        .AddOutput(*offset_out)
        .AddAttr("ksize", kernel_size)
        .AddAttr("strides", strides_offset)
        .AddAttr("pads", pads_offset)
        .AddAttr("dilations", dilations_offset)
        .AddAttr("deformable_groups", deformable_groups)
        .AddAttr("data_format", data_layout)
        .AddAttr("modulated", true)
        .Run(stream);

    PrintTensor(offset_out, "offset_out", "Output - forward");

    // LOG(INFO) << "offset_output shape = " << offset_output.dims().to_str();

    // Conv2D
    // const std::vector<int> strides_conv2d = {1, 1};
    const std::vector<int64_t> strides_conv2d = {1, 1, kernel_size[0], kernel_size[1]};
    const std::vector<int64_t> paddings_conv2d = {0, 0, 0, 0};
    const std::vector<int64_t> dilations_conv2d = {1, 1, 1, 1};

    NpuOpRunner runner_conv;
    runner_conv.SetType("Conv2D")
        .AddInput(*offset_out)
        .AddInput(*filter)
        .AddOutput(*output)
        .AddAttr("strides", strides_conv2d)
        .AddAttr("pads", paddings_conv2d)
        .AddAttr("dilations", dilations_conv2d)
        .AddAttr("groups", groups)
        .AddAttr("data_format", data_layout)
        .Run(stream);

    PrintTensor(output, "output", "Output - forward");
  }
};

template <typename T>
class DeformableConvGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor* offset_grad = ctx.Output<Tensor>(framework::GradVarName("Offset"));
    Tensor* mask_grad = ctx.Output<Tensor>(framework::GradVarName("Mask"));

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* offset_out = ctx.Input<Tensor>("OffsetOut");
    const Tensor* offset = ctx.Input<Tensor>("Offset");
    const Tensor* mask = ctx.Input<Tensor>("Mask");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    if (!input_grad && !filter_grad && !offset_grad && !mask_grad) return;

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    // const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::vector<int64_t> kernel_size{filter->dims()[2], filter->dims()[3]};
    const std::string data_layout = framework::DataLayoutToString(input->layout());

    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    PrintTensor(input, "input", "Input - backward");
    PrintTensor(offset, "offset", "Input - backward");
    PrintTensor(filter, "filter", "Input - backward");
    PrintTensor(offset_out, "offset_out", "Input - backward");
    PrintTensor(output_grad, "output_grad", "Input - backward");
    
    const std::vector<int64_t> strides_conv2d = {1, 1, kernel_size[0], kernel_size[1]};
    const std::vector<int64_t> paddings_conv2d = {0, 0, 0, 0};
    const std::vector<int64_t> dilations_conv2d = {1, 1, 1, 1};

    if(filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_conv_grad;
      runner_conv_grad.SetType("Conv2DBackpropFilterD")
          .AddInput(*offset_out)
          .AddInput(*output_grad)
          .AddOutput(*filter_grad)
          .AddAttr("filter_size", framework::vectorize<int64_t>(filter->dims()))
          .AddAttr("strides", strides_conv2d)
          .AddAttr("pads", paddings_conv2d)
          .AddAttr("dilations", dilations_conv2d)
          .AddAttr("groups", groups)
          .AddAttr("data_format", data_layout)
          .Run(stream);
      PrintTensor(filter_grad, "filter_grad", "Output - backward");
    }
    if (input_grad || offset_grad || mask_grad) {
      Tensor input_grad_tmp(input->type());
      if (input_grad == nullptr) {
        input_grad_tmp.Resize(input->dims());
        input_grad = &input_grad_tmp;
      }
      input_grad->mutable_data<T>(ctx.GetPlace());

      auto offset_out_grad = ctx.AllocateTmpTensor<T, NPUDeviceContext>(offset_out->dims(), dev_ctx);
      NpuOpRunner runner_conv_grad;
      runner_conv_grad.SetType("Conv2DBackpropInputD")
          .AddInput(*filter)
          .AddInput(*output_grad)
          .AddOutput(offset_out_grad)
          .AddAttr("input_size", framework::vectorize<int64_t>(offset_out->dims()))
          .AddAttr("strides", strides_conv2d)
          .AddAttr("pads", paddings_conv2d)
          .AddAttr("dilations", dilations_conv2d)
          .AddAttr("groups", groups)
          .AddAttr("data_format", data_layout)
          .Run(stream);

      const std::vector<int64_t> offset_concat_size{offset->dims()[0], 3 * kernel_size[0] * kernel_size[1], offset->dims()[2], offset->dims()[3]};
      auto offset_concat = ctx.AllocateTmpTensor<T, NPUDeviceContext>(framework::make_ddim(offset_concat_size), dev_ctx);
      NpuOpRunner runner_concat;
      runner_concat.SetType("ConcatD")
          .AddInput(*offset)
          .AddInput(*mask)
          .AddOutput(offset_concat)
          .AddAttr("concat_dim", 1)
          .AddAttr("N", 2)
          .AddInputNames({"x0", "x1"})
          .Run(stream);
      PrintTensor(&offset_concat, "offset_concat", "Temp - backward");

      auto offset_concat_grad = ctx.AllocateTmpTensor<T, NPUDeviceContext>(framework::make_ddim(offset_concat_size), dev_ctx);
      const std::vector<int64_t> strides_offset = {1, 1, strides[0], strides[1]};
      const std::vector<int64_t> pads_offset{paddings[0], paddings[0], paddings[1], paddings[1]};
      const std::vector<int64_t> dilations_offset = {1, 1, dilations[0], dilations[1]};
      NpuOpRunner runner_offset_grad;
      runner_offset_grad.SetType("DeformableOffsetsGrad")
          .AddInput(offset_out_grad)
          .AddInput(*input)
          .AddInput(offset_concat)
          .AddOutput(*input_grad)
          .AddOutput(offset_concat_grad)
          .AddAttr("strides", strides_offset)
          .AddAttr("pads", pads_offset)
          .AddAttr("ksize", kernel_size)
          .AddAttr("dilations", dilations_offset)
          .AddAttr("data_format", data_layout)
          .AddAttr("deformable_groups", deformable_groups)
          .AddAttr("modulated", true)
          .Run(stream);

      PrintTensor(&offset_concat_grad, "offset_concat_grad", "Temp - backward");

      if (offset_grad || mask_grad) {
        Tensor offset_grad_tmp(filter->type());
        Tensor mask_grad_tmp(mask->type());
        if (offset_grad == nullptr) {
          offset_grad_tmp.Resize(filter->dims());
          offset_grad = &offset_grad_tmp;
        }
        if (mask_grad == nullptr) {
          mask_grad_tmp.Resize(mask->dims());
          mask_grad = &mask_grad_tmp;
        }
        offset_grad->mutable_data<T>(ctx.GetPlace());
        mask_grad->mutable_data<T>(ctx.GetPlace());

        std::vector<Tensor> outputs{*offset_grad, *mask_grad};
        const std::vector<int64_t> sections{2 * kernel_size[0] * kernel_size[1], kernel_size[0] * kernel_size[1]};
        NpuOpRunner runner_split;
        runner_split.SetType("SplitVD")
            .AddInput(offset_concat_grad)
            .AddOutputs(outputs)
            .AddAttr("size_splits", sections)
            .AddAttr("split_dim", 1)
            .AddAttr("num_split", 2)
            .Run(stream);
        PrintTensor(offset_grad, "offset_grad", "Output - backward");
        PrintTensor(mask_grad, "mask_grad", "Output - backward");
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(deformable_conv,
                       paddle::operators::DeformableConvNPUKernel<float>);

REGISTER_OP_NPU_KERNEL(deformable_conv_grad,
                       paddle::operators::DeformableConvGradNPUKernel<float>);
