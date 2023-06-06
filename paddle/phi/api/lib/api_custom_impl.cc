/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/lib/api_custom_impl.h"

#include "glog/logging.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/debug_op.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/tensor_copy.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace experimental {

////////////////// Forward api impls //////////////////////

Tensor add_n_impl(const std::vector<Tensor>& x) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  bool is_sr_kernel = true;
  for (auto& input : x) {
    if (phi::DenseTensor::classof(input.impl().get())) {
      is_sr_kernel = false;
      break;
    }
  }

  const std::string kernel_name = (is_sr_kernel ? "add_n_sr" : "add_n");

  VLOG(6) << "add_n API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << kernel_name << " kernel: " << kernel;
  auto* dev_ctx = GetDeviceContextByBackend(
      kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  Tensor api_output;

  if (is_sr_kernel) {
    std::vector<const phi::SelectedRows*> input_x(x.size());
    for (size_t i = 0; i < input_x.size(); ++i) {
      input_x[i] = static_cast<phi::SelectedRows*>(x[i].impl().get());
    }
    auto x_meta_vec = MakeMetaTensor(input_x);
    std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
    for (size_t i = 0; i < x_meta_vec.size(); ++i) {
      x_metas[i] = &x_meta_vec[i];
    }
    auto kernel_out = SetSelectedRowsKernelOutput(&api_output);
    phi::MetaTensor meta_out(kernel_out);
    phi::AddNInferMeta(x_metas, &meta_out);

    using kernel_signature =
        void (*)(const phi::DeviceContext&,
                 const std::vector<const phi::SelectedRows*>&,
                 phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  } else {
    std::vector<const phi::TensorBase*> input_x(x.size());
    std::vector<std::shared_ptr<phi::DenseTensor>> temp_dense_tensots;
    temp_dense_tensots.reserve(x.size());
    for (size_t i = 0; i < input_x.size(); ++i) {
      if (phi::DenseTensor::classof(x[i].impl().get())) {
        temp_dense_tensots.push_back(PrepareData(x[i], kernel.InputAt(0), {}));
        input_x[i] = temp_dense_tensots.back().get();
      } else {
        input_x[i] = x[i].impl().get();
      }
    }
    auto x_meta_vec = MakeMetaTensor(input_x);
    std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
    for (size_t i = 0; i < x_meta_vec.size(); ++i) {
      x_metas[i] = &x_meta_vec[i];
    }
    auto kernel_out = SetKernelOutput(&api_output);
    phi::MetaTensor meta_out(kernel_out);
    phi::AddNInferMeta(x_metas, &meta_out);

    using kernel_signature =
        void (*)(const phi::DeviceContext&,
                 const std::vector<const phi::TensorBase*>&,
                 phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

    paddle::experimental::OpIdAdd();
    VLOG(10) << "Op ID: " << paddle::experimental::OpId();
    std::string debug_str = "";
    std::vector<const phi::TensorBase*> dev2_input_x(x.size());
    std::vector<std::shared_ptr<phi::DenseTensor>> dev2_temp_dense_tensots;
    std::vector<std::shared_ptr<phi::SelectedRows>> dev2_temp_select_rows;
    dev2_temp_dense_tensots.reserve(x.size());
    dev2_temp_select_rows.reserve(x.size());
    std::shared_ptr<phi::DenseTensor> dev2_kernel_out_smart_ptr;
    phi::DenseTensor* dev2_kernel_out = nullptr;
    if (paddle::experimental::DebugOrNot() && ContinueOrNot("add_n")) {
      VLOG(10) << "Start copy input and output!";
      for (size_t i = 0; i < input_x.size(); ++i) {
        if (phi::DenseTensor::classof(input_x[i])) {
          dev2_temp_dense_tensots.push_back(
              paddle::experimental::CopyDenseTensor(
                  static_cast<const phi::DenseTensor*>(input_x[i]),
                  paddle::experimental::GetDebugDev2Type()));
          dev2_input_x[i] = dev2_temp_dense_tensots.back().get();
        } else if (phi::SelectedRows::classof(input_x[i])) {
          dev2_temp_select_rows.push_back(
              paddle::experimental::CopySelectedRows(
                  static_cast<const phi::SelectedRows*>(input_x[i]),
                  paddle::experimental::GetDebugDev2Type()));
          dev2_input_x[i] = dev2_temp_select_rows.back().get();
        } else {
          PADDLE_THROW(phi::errors::InvalidArgument(
              "Expected type of Input(X) of %d-th must be Tensor, "
              "SelectedRows. But got "
              "unsupport type: %s.",
              input_x[i]->type_info().name()));
        }
      }
      dev2_kernel_out_smart_ptr = paddle::experimental::CopyDenseTensor(
          kernel_out, paddle::experimental::GetDebugDev2Type());
      dev2_kernel_out =
          dev2_kernel_out_smart_ptr ? dev2_kernel_out_smart_ptr.get() : nullptr;
      VLOG(10) << "End copy input and output!";
      VLOG(10) << "Start check acc for input!";
      debug_str += paddle::experimental::XPUDebugString(
          "add_n", "input_x", input_x, dev2_input_x);
      VLOG(10) << "End check acc for input!";
    }
    (*kernel_fn)(*dev_ctx, input_x, kernel_out);

    if (paddle::experimental::DebugOrNot() && ContinueOrNot("add_n")) {
      Backend dev2_kernel_backend = TransToPhiBackend(GetDebugDev2Type());
      VLOG(6) << "add_n API kernel key Dev2: [" << dev2_kernel_backend << ", "
              << kernel_layout << ", " << kernel_data_type << "]";
      auto dev2_kernel_result =
          phi::KernelFactory::Instance().SelectKernelOrThrowError(
              "add_n", {dev2_kernel_backend, kernel_layout, kernel_data_type});
      const auto& dev2_kernel = dev2_kernel_result.kernel;
      auto* dev2_ctx = GetDeviceContextByBackend(
          dev2_kernel_result.has_fallback_cpu ? Backend::CPU
                                              : dev2_kernel_backend);
      auto* dev2_kernel_fn =
          dev2_kernel.GetVariadicKernelFn<kernel_signature>();
      std::string debug_start_str = paddle::experimental::XPUDebugStartString(
          "add_n",
          kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend,
          kernel_data_type);
      VLOG(10) << "Strat run kernel on device 2";
      (*dev2_kernel_fn)(*dev2_ctx, dev2_input_x, dev2_kernel_out);
      VLOG(10) << "End run kernel on device 2";
      VLOG(10) << "Start check acc for output!";
      debug_str += "out: ";
      debug_str += paddle::experimental::XPUDebugString(
          "add_n", "kernel_out", kernel_out, dev2_kernel_out);
      VLOG(10) << "End check acc for output!";
      if (debug_str != "out: ")
        std::cout << debug_start_str << "in: " << debug_str << std::endl;
    }
    if (kernel_result.has_fallback_cpu) {
      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }
  }

  return api_output;
}

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking) {
  Tensor out;
  copy(x, place, blocking, &out);
  out.set_name(x.name());
  return out;
}

////////////////// Backward(grad) api impls //////////////////////

void embedding_grad_impl(const Tensor& x,
                         const Tensor& weight,
                         const Tensor& out_grad,
                         int64_t padding_idx,
                         bool sparse,
                         Tensor* weight_grad) {
  DataType kernel_data_type = ParseDataType(weight);
  auto kernel_key_set = ParseKernelKeyByInputArgs(weight);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  VLOG(6) << "embedding_grad API kernel key: [" << kernel_key.backend() << ", "
          << kernel_key.layout() << ", " << kernel_data_type << "]";

  if (phi::DenseTensor::classof(weight.impl().get())) {
    std::string kernel_name =
        sparse ? "embedding_sparse_grad" : "embedding_grad";
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(
        kernel_result.has_fallback_cpu ? Backend::CPU : kernel_key.backend());

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = PrepareData(weight, kernel.InputAt(1), {});
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out = SetSelectedRowsKernelOutput(weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      meta_out.set_dims(input_weight->dims());
      meta_out.set_dtype(input_weight->dtype());
      kernel_out->set_height(input_weight->dims()[0]);

      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const phi::DenseTensor&,
                                        const phi::DenseTensor&,
                                        int64_t,
                                        phi::SelectedRows*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx,
                   *input_x,
                   *input_weight,
                   *input_out_grad,
                   padding_idx,
                   kernel_out);
    } else {
      auto* kernel_out = SetKernelOutput(weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      phi::UnchangedInferMeta(MakeMetaTensor(*input_weight), &meta_out);
      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const phi::DenseTensor&,
                                        const phi::DenseTensor&,
                                        int64_t,
                                        phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

      paddle::experimental::OpIdAdd();
      VLOG(10) << "Op ID: " << paddle::experimental::OpId();
      std::string debug_str = "";
      std::shared_ptr<phi::DenseTensor> dev2_input_x;
      std::shared_ptr<phi::DenseTensor> dev2_input_weight;
      std::shared_ptr<phi::DenseTensor> dev2_input_out_grad;
      std::shared_ptr<phi::DenseTensor> dev2_kernel_out_smart_ptr;
      phi::DenseTensor* dev2_kernel_out = nullptr;
      if (paddle::experimental::DebugOrNot() &&
          ContinueOrNot("embedding_grad")) {
        VLOG(10) << "Start copy input and output!";
        dev2_input_x = paddle::experimental::CopyDenseTensor(
            input_x, paddle::experimental::GetDebugDev2Type());
        dev2_input_weight = paddle::experimental::CopyDenseTensor(
            input_weight, paddle::experimental::GetDebugDev2Type());
        dev2_input_out_grad = paddle::experimental::CopyDenseTensor(
            input_out_grad, paddle::experimental::GetDebugDev2Type());

        dev2_kernel_out_smart_ptr = paddle::experimental::CopyDenseTensor(
            kernel_out, paddle::experimental::GetDebugDev2Type());
        dev2_kernel_out = dev2_kernel_out_smart_ptr
                              ? dev2_kernel_out_smart_ptr.get()
                              : nullptr;
        VLOG(10) << "End copy input and output!";
        VLOG(10) << "Start check acc for input!";
        debug_str += paddle::experimental::XPUDebugString(
            "embedding_grad", "input_x;", *input_x, *dev2_input_x);
        debug_str += paddle::experimental::XPUDebugString("embedding_grad",
                                                          "input_weight",
                                                          *input_weight,
                                                          *dev2_input_weight);
        debug_str += paddle::experimental::XPUDebugString("embedding_grad",
                                                          "input_out_grad",
                                                          *input_out_grad,
                                                          *dev2_input_out_grad);
        VLOG(10) << "End check acc for input!";
      }
      (*kernel_fn)(*dev_ctx,
                   *input_x,
                   *input_weight,
                   *input_out_grad,
                   padding_idx,
                   kernel_out);
      if (paddle::experimental::DebugOrNot() &&
          ContinueOrNot("embedding_grad")) {
        Backend dev2_kernel_backend = TransToPhiBackend(GetDebugDev2Type());
        VLOG(6) << "embedding_grad API kernel key Dev2: ["
                << dev2_kernel_backend << ", " << kernel_key.layout() << ", "
                << kernel_data_type << "]";
        auto dev2_kernel_result =
            phi::KernelFactory::Instance().SelectKernelOrThrowError(
                "embedding_grad",
                {dev2_kernel_backend, kernel_key.layout(), kernel_data_type});
        const auto& dev2_kernel = dev2_kernel_result.kernel;
        auto* dev2_ctx = GetDeviceContextByBackend(
            dev2_kernel_result.has_fallback_cpu ? Backend::CPU
                                                : dev2_kernel_backend);
        auto* dev2_kernel_fn =
            dev2_kernel.GetVariadicKernelFn<kernel_signature>();
        std::string debug_start_str = paddle::experimental::XPUDebugStartString(
            "embedding_grad",
            kernel_result.has_fallback_cpu ? Backend::CPU
                                           : kernel_key.backend(),
            kernel_data_type);
        VLOG(10) << "Strat run kernel on device 2";
        (*dev2_kernel_fn)(*dev2_ctx,
                          *dev2_input_x,
                          *dev2_input_weight,
                          *dev2_input_out_grad,
                          padding_idx,
                          dev2_kernel_out);
        VLOG(10) << "End run kernel on device 2";
        VLOG(10) << "Start check acc for output!";
        debug_str += "out: ";
        debug_str += paddle::experimental::XPUDebugString(
            "embedding_grad", "kernel_out", kernel_out, dev2_kernel_out);
        VLOG(10) << "End check acc for output!";
        if (debug_str != "out: ")
          std::cout << debug_start_str << "in: " << debug_str << std::endl;
      }
    }
  } else {
    std::string kernel_name = sparse ? "sparse_weight_embedding_sparse_grad"
                                     : "sparse_weight_embedding_grad";
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(
        kernel_result.has_fallback_cpu ? Backend::CPU : kernel_key.backend());

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = TensorToSelectedRows(weight);
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out = SetSelectedRowsKernelOutput(weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      phi::UnchangedInferMeta(MakeMetaTensor(*input_weight), &meta_out);
      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const phi::SelectedRows&,
                                        const phi::DenseTensor&,
                                        int64_t,
                                        phi::SelectedRows*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx,
                   *input_x,
                   *input_weight,
                   *input_out_grad,
                   padding_idx,
                   kernel_out);
    } else {
      auto* kernel_out = SetKernelOutput(weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      meta_out.set_dims(input_weight->GetCompleteDims());
      meta_out.set_dtype(input_weight->dtype());
      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const phi::SelectedRows&,
                                        const phi::DenseTensor&,
                                        int64_t,
                                        phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx,
                   *input_x,
                   *input_weight,
                   *input_out_grad,
                   padding_idx,
                   kernel_out);
    }
  }
}

}  // namespace experimental
}  // namespace paddle
