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

#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/context_project.h"
#include "paddle/fluid/operators/math/math_function.h"

#define CLOG std::cout

namespace paddle {
namespace operators {

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(const char)};
  framework::LoD lod;
  int summarize;
  void* data{nullptr};

  void operator()(size_t size) {
    // PrintMessage();
    // PrintName();
    // PrintDims();
    // PrintDtype();
    // PrintLod();
    PrintData(size);
  }

 private:
  void PrintMessage() { CLOG << std::time(nullptr) << "\t" << message << "\t"; }
  void PrintName() {
    if (!name.empty()) {
      CLOG << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      CLOG << "\tshape: [";
      for (auto i : dims) {
        CLOG << i << ",";
      }
      CLOG << "]" << std::endl;
    }
  }
  void PrintDtype() {
    if (dtype.hash_code() != typeid(const char).hash_code()) {
      CLOG << "\tdtype: " << dtype.name() << std::endl;
    }
  }
  void PrintLod() {
    if (!lod.empty()) {
      CLOG << "\tLoD: [";
      for (auto level : lod) {
        CLOG << "[ ";
        for (auto i : level) {
          CLOG << i << ",";
        }
        CLOG << " ]";
      }
      CLOG << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (dtype.hash_code() == typeid(const float).hash_code()) {
      Display<float>(size);
    } else if (dtype.hash_code() == typeid(const double).hash_code()) {
      Display<double>(size);
    } else if (dtype.hash_code() == typeid(const int).hash_code()) {
      Display<int>(size);
    } else if (dtype.hash_code() == typeid(const int64_t).hash_code()) {
      Display<int64_t>(size);
    } else if (dtype.hash_code() == typeid(const bool).hash_code()) {
      Display<bool>(size);
    } else {
      CLOG << "\tdata: unprintable type: " << dtype.name() << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto* d = reinterpret_cast<T*>(data);
    CLOG << "\tdata: " << size << std::endl;
    if (summarize != -1) {
      summarize = 10000;
      CLOG << "Value of summarize = " << summarize << std::endl;
      for (int i = 0; i < summarize; i++) {
        CLOG << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        CLOG << d[i] << ",";
      }
    }
    CLOG << std::endl;
  }
};

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto filter = *context.Input<Tensor>("Filter");

    out->mutable_data<T>(context.GetPlace());

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");

    const Tensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<Tensor>("PaddingData");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width = static_cast<int>(in->dims()[1]);

    framework::DDim col_shape = {in->dims()[0],
                                 context_length * sequence_width};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // Because if padding_trainable is false, padding data should be zeros.
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    set_zero(dev_ctx, &col, static_cast<T>(0));
    math::ContextProjectFunctor<DeviceContext, T> seq_project_functor;

    seq_project_functor(dev_ctx, *in, *padding_data, padding_trainable,
                        context_start, context_length, context_stride, up_pad,
                        down_pad, &col);

    blas.MatMul(col, filter, out);
  }
};

static void print_sid_stuff(const framework::LoDTensor* in_g) {
  framework::LoDTensor print_tensor_din;
  print_tensor_din.Resize(in_g->dims());
  std::cout << print_tensor_din.dims() << std::endl;

  if (paddle::platform::is_cpu_place(in_g->place())) {
    print_tensor_din.ShareDataWith(*in_g);
  } else {
    // copy data to cpu to print
    std::cout << "Should be printed, part 2" << std::endl;
    paddle::platform::CPUPlace place;
    framework::TensorCopy(*in_g, place, &print_tensor_din);
  }
  Formater formater5;
  formater5.dtype = print_tensor_din.type();
  formater5.data = reinterpret_cast<void*>(print_tensor_din.data<void>());
  formater5(print_tensor_din.numel());
}

template <typename DeviceContext, typename T>
class SequenceConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* filter_g = context.Output<Tensor>(framework::GradVarName("Filter"));
    auto* padding_data_g =
        context.Output<Tensor>(framework::GradVarName("PaddingData"));
    auto* in = context.Input<LoDTensor>("X");
    auto* filter = context.Input<Tensor>("Filter");

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in->lod()[0];

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width = static_cast<int>(in->dims()[1]);

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    // use col_shape in the im2col calculation
    framework::DDim col_shape = {in->dims()[0],
                                 sequence_width * context_length};
    Tensor col;

    if (in_g || filter_g || (padding_trainable && padding_data_g)) {
      col.mutable_data<T>(col_shape, context.GetPlace());
      // Because if padding_trainable is false, padding data should be zeros.
      set_zero(dev_ctx, &col, static_cast<T>(0));
      blas.MatMul(*out_g, false, *filter, true, &col);
    }
    math::ContextProjectFunctor<DeviceContext, T> seq_project_functor;
    math::ContextProjectGradFunctor<DeviceContext, T> seq_project_grad_functor;

    if (in_g) {
      in_g->mutable_data<T>(context.GetPlace());
      in_g->set_lod(in->lod());
      set_zero(dev_ctx, in_g, static_cast<T>(0));

      seq_project_grad_functor(dev_ctx, *in_g, padding_trainable, context_start,
                               context_length, context_stride, up_pad, down_pad,
                               false, true, padding_data_g, &col);

      std::cout << "Will print in_g (which is the output) in sequence_conv"
                << std::endl;

      print_sid_stuff(in_g);
      std::cout << "Will print in (which is input) in sequence_conv"
                << std::endl;
      print_sid_stuff(in);

      std::cout << "Will print out_g (which is input) in sequence_conv"
                << std::endl;
      print_sid_stuff(out_g);
      /*
            framework::LoDTensor print_tensor_din;
            print_tensor_din.Resize(in_g->dims());
            std::cout << "Printing d_input" << std::endl;
            std::cout << print_tensor_din.dims() << std::endl;

            if (paddle::platform::is_cpu_place(in_g->place())) {
              print_tensor_din.ShareDataWith(*in_g);
            } else {
              // copy data to cpu to print
              std::cout << "Should be printed, part 2" << std::endl;
              paddle::platform::CPUPlace place;
              framework::TensorCopy(*in_g, place, &print_tensor_din);
            }
            Formater formater5;
            formater5.dtype = print_tensor_din.type();
            formater5.data =
         reinterpret_cast<void*>(print_tensor_din.data<void>());
            formater5(print_tensor_din.numel());
      */
    }

    if (padding_trainable && padding_data_g) {
      padding_data_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, padding_data_g, static_cast<T>(0));

      LoDTensor* input = const_cast<LoDTensor*>(in);
      seq_project_grad_functor(
          dev_ctx, *input, padding_trainable, context_start, context_length,
          context_stride, up_pad, down_pad, true, false, padding_data_g, &col);
    }

    if (filter_g) {
      filter_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_g, static_cast<T>(0));

      Tensor filter_grad = *filter_g;
      LoDTensor out_grad = *out_g;

      const Tensor* padding_data = nullptr;
      if (padding_trainable) {
        padding_data = context.Input<Tensor>("PaddingData");
      }

      seq_project_functor(dev_ctx, *in, *padding_data, padding_trainable,
                          context_start, context_length, context_stride, up_pad,
                          down_pad, &col);

      blas.MatMul(col, true, out_grad, false, &filter_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
