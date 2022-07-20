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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> adamw_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& learning_rate,
    const Tensor& moment1,
    const Tensor& moment2,
    const Tensor& beta1_pow,
    const Tensor& beta2_pow,
    const paddle::optional<Tensor>& master_param,
    const paddle::optional<Tensor>& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    float lr_ratio,
    float coeff,
    bool with_decay,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param);
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
  std::string kernel_name = "adamw";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << kernel_name << " API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << kernel_name << " API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_param = PrepareData(param, kernel.InputAt(0), {});
  auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
  auto input_lr = PrepareData(learning_rate, kernel.InputAt(2), {});
  auto input_moment1 = PrepareData(moment1, kernel.InputAt(3), {});
  auto input_moment2 = PrepareData(moment2, kernel.InputAt(4), {});
  auto input_beta1_pow = PrepareData(beta1_pow, kernel.InputAt(5), {});
  auto input_beta2_pow = PrepareData(beta2_pow, kernel.InputAt(6), {});
  auto input_master_param = PrepareData(master_param, kernel.InputAt(7), {});
  auto input_skip_update = PrepareData(skip_update, kernel.InputAt(8), {});

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = input_param.get();
  auto kernel_out_1 = input_moment1.get();
  auto kernel_out_2 = input_moment2.get();
  auto kernel_out_3 = input_beta1_pow.get();
  auto kernel_out_4 = input_beta2_pow.get();
  phi::DenseTensor* kernel_out_5 = nullptr;
  if (input_master_param) {
    kernel_out_5 = input_master_param.get_ptr();
  }

  auto input_meta_ref_master_param = MakeMetaTensor(input_master_param);

  auto input_meta_ref_skip_update = MakeMetaTensor(input_skip_update);

  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);
  phi::MetaTensor meta_out_4(kernel_out_4);
  phi::MetaTensor meta_out_5(kernel_out_5);

  phi::AdamwInferMeta(MakeMetaTensor(*input_param),
                      MakeMetaTensor(*input_grad),
                      MakeMetaTensor(*input_lr),
                      MakeMetaTensor(*input_moment1),
                      MakeMetaTensor(*input_moment2),
                      MakeMetaTensor(*input_beta1_pow),
                      MakeMetaTensor(*input_beta2_pow),
                      input_meta_ref_master_param,
                      input_meta_ref_skip_update,
                      beta1,
                      beta2,
                      epsilon,
                      lr_ratio,
                      coeff,
                      with_decay,
                      lazy_mode,
                      min_row_size_to_use_multithread,
                      multi_precision,
                      use_global_beta_pow,
                      &meta_out_0,
                      &meta_out_1,
                      &meta_out_2,
                      &meta_out_3,
                      &meta_out_4,
                      &meta_out_5);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const paddle::optional<phi::DenseTensor>&,
                                    const paddle::optional<phi::DenseTensor>&,
                                    const Scalar&,
                                    const Scalar&,
                                    const Scalar&,
                                    float,
                                    float,
                                    bool,
                                    bool,
                                    int64_t,
                                    bool,
                                    bool,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  (*kernel_fn)(*dev_ctx,
               *input_param,
               *input_grad,
               *input_lr,
               *input_moment1,
               *input_moment2,
               *input_beta1_pow,
               *input_beta2_pow,
               input_master_param,
               input_skip_update,
               beta1,
               beta2,
               epsilon,
               lr_ratio,
               coeff,
               with_decay,
               lazy_mode,
               min_row_size_to_use_multithread,
               multi_precision,
               use_global_beta_pow,
               kernel_out_0,
               kernel_out_1,
               kernel_out_2,
               kernel_out_3,
               kernel_out_4,
               kernel_out_5);

  return api_output;
}

Tensor conv2d_impl(const Tensor& input,
                   const Tensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& paddding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   bool use_addto,
                   int workspace_size_MB,
                   bool exhaustive_search) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(input);

  VLOG(6) << "conv2d API kernel key: [" << input.layout() << ", "
          << kernel_layout << ", " << filter.layout() << "]";
  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, filter);
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

  VLOG(6) << "conv2d API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv2d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "conv2d API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  phi::TensorArgDef args0 = kernel.InputAt(0);
  phi::TensorArgDef args1 = kernel.InputAt(1);
  if (kernel_backend == Backend::GPU) {
    args0.backend = Backend::GPU;
    args1.backend = Backend::GPU;
    args0.layout = input.layout();
    args1.layout = filter.layout();
  }

  auto input_input = PrepareData(input, args0, {});
  auto input_filter = PrepareData(filter, args1, {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvInferMeta(MakeMetaTensor(*input_input),
                     MakeMetaTensor(*input_filter),
                     strides,
                     paddings,
                     paddding_algorithm,
                     groups,
                     dilations,
                     data_format,
                     use_addto,
                     workspace_size_MB,
                     exhaustive_search,
                     &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const std::vector<int>&,
                                    const std::vector<int>&,
                                    const std::string&,
                                    int,
                                    const std::vector<int>&,
                                    const std::string&,
                                    bool,
                                    int,
                                    bool,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  {
    (*kernel_fn)(*dev_ctx,
                 *input_input,
                 *input_filter,
                 strides,
                 paddings,
                 paddding_algorithm,
                 groups,
                 dilations,
                 data_format,
                 use_addto,
                 workspace_size_MB,
                 exhaustive_search,
                 kernel_out);
  }

  return api_output;
}

Tensor conv3d_impl(const Tensor& input,
                   const Tensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& paddding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   bool use_addto,
                   int workspace_size_MB,
                   bool exhaustive_search) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(input);

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, filter);
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

  VLOG(6) << "conv3d API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv3d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "conv3d API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  phi::TensorArgDef args0 = kernel.InputAt(0);
  phi::TensorArgDef args1 = kernel.InputAt(1);
  if (kernel_backend == Backend::GPU) {
    args0.backend = Backend::GPU;
    args1.backend = Backend::GPU;
  }

  auto input_input = PrepareData(input, args0, {});
  auto input_filter = PrepareData(filter, args1, {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvInferMeta(MakeMetaTensor(*input_input),
                     MakeMetaTensor(*input_filter),
                     strides,
                     paddings,
                     paddding_algorithm,
                     groups,
                     dilations,
                     data_format,
                     use_addto,
                     workspace_size_MB,
                     exhaustive_search,
                     &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const std::vector<int>&,
                                    const std::vector<int>&,
                                    const std::string&,
                                    int,
                                    const std::vector<int>&,
                                    const std::string&,
                                    bool,
                                    int,
                                    bool,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  {
    (*kernel_fn)(*dev_ctx,
                 *input_input,
                 *input_filter,
                 strides,
                 paddings,
                 paddding_algorithm,
                 groups,
                 dilations,
                 data_format,
                 use_addto,
                 workspace_size_MB,
                 exhaustive_search,
                 kernel_out);
  }

  return api_output;
}

void conv2d_grad_impl(const Tensor& input,
                      const Tensor& filter,
                      const Tensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& paddding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      bool use_addto,
                      int workspace_size_MB,
                      bool exhaustive_search,
                      Tensor* input_grad,
                      Tensor* filter_grad) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, filter, out_grad);
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

  VLOG(6) << "conv2d_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv2d_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "conv2d_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  phi::TensorArgDef args0 = kernel.InputAt(0);
  phi::TensorArgDef args1 = kernel.InputAt(1);
  phi::TensorArgDef args2 = kernel.InputAt(2);
  if (kernel_backend == Backend::GPU) {
    args0.backend = Backend::GPU;
    args1.backend = Backend::GPU;
    args2.backend = Backend::GPU;
  }

  auto input_input = PrepareData(input, args0, {});
  auto input_filter = PrepareData(filter, args1, {});
  auto input_out_grad = PrepareData(out_grad, args2, {});

  auto kernel_out_0 = SetKernelOutput(kernel_backend, input_grad);
  auto kernel_out_1 = SetKernelOutput(kernel_backend, filter_grad);
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_input),
                                  MakeMetaTensor(*input_filter),
                                  kernel_out_0 ? &meta_out_0 : nullptr,
                                  kernel_out_1 ? &meta_out_1 : nullptr);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const std::vector<int>&,
                                    const std::vector<int>&,
                                    const std::string&,
                                    int,
                                    const std::vector<int>&,
                                    const std::string&,
                                    bool,
                                    int,
                                    bool,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  {
    (*kernel_fn)(*dev_ctx,
                 *input_input,
                 *input_filter,
                 *input_out_grad,
                 strides,
                 paddings,
                 paddding_algorithm,
                 groups,
                 dilations,
                 data_format,
                 use_addto,
                 workspace_size_MB,
                 exhaustive_search,
                 kernel_out_0,
                 kernel_out_1);
  }
}

void conv3d_grad_impl(const Tensor& input,
                      const Tensor& filter,
                      const Tensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& paddding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      bool use_addto,
                      int workspace_size_MB,
                      bool exhaustive_search,
                      Tensor* input_grad,
                      Tensor* filter_grad) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, filter, out_grad);
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

  VLOG(6) << "conv3d_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv3d_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "conv3d_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  phi::TensorArgDef args0 = kernel.InputAt(0);
  phi::TensorArgDef args1 = kernel.InputAt(1);
  phi::TensorArgDef args2 = kernel.InputAt(2);
  if (kernel_backend == Backend::GPU) {
    args0.backend = Backend::GPU;
    args1.backend = Backend::GPU;
    args2.backend = Backend::GPU;
  }

  auto input_input = PrepareData(input, args0, {});
  auto input_filter = PrepareData(filter, args1, {});
  auto input_out_grad = PrepareData(out_grad, args2, {});

  auto kernel_out_0 = SetKernelOutput(kernel_backend, input_grad);
  auto kernel_out_1 = SetKernelOutput(kernel_backend, filter_grad);
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_input),
                                  MakeMetaTensor(*input_filter),
                                  kernel_out_0 ? &meta_out_0 : nullptr,
                                  kernel_out_1 ? &meta_out_1 : nullptr);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const std::vector<int>&,
                                    const std::vector<int>&,
                                    const std::string&,
                                    int,
                                    const std::vector<int>&,
                                    const std::string&,
                                    bool,
                                    int,
                                    bool,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  {
    (*kernel_fn)(*dev_ctx,
                 *input_input,
                 *input_filter,
                 *input_out_grad,
                 strides,
                 paddings,
                 paddding_algorithm,
                 groups,
                 dilations,
                 data_format,
                 use_addto,
                 workspace_size_MB,
                 exhaustive_search,
                 kernel_out_0,
                 kernel_out_1);
  }
}

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking) {
  Tensor out;
  copy(x, place, blocking, &out);
  return out;
}

Tensor embedding_impl(const Tensor& x,
                      const Tensor& weight,
                      int64_t padding_idx,
                      bool sparse) {
  DataType kernel_data_type = ParseDataType(weight);
  auto kernel_key_set = ParseKernelKeyByInputArgs(weight);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  VLOG(6) << "embedding API kernel key: [" << kernel_key.backend() << ", "
          << kernel_key.layout() << ", " << kernel_data_type << "]";

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  Tensor api_output;

  if (phi::DenseTensor::classof(weight.impl().get())) {
    const auto& kernel =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            "embedding",
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    VLOG(6) << "embedding API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = PrepareData(weight, kernel.InputAt(1), {});

    auto* kernel_out = SetKernelOutput(kernel_key.backend(), &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::EmbeddingInferMeta(MakeMetaTensor(*input_x),
                            MakeMetaTensor(*input_weight),
                            padding_idx,
                            sparse,
                            &meta_out);

    using kernel_signature = void (*)(const platform::DeviceContext&,
                                      const phi::DenseTensor&,
                                      const phi::DenseTensor&,
                                      int64_t,
                                      phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      (*kernel_fn)(*dev_ctx, *input_x, *input_weight, padding_idx, kernel_out);
    }
  } else {
    const auto& kernel =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            "sparse_weight_embedding",
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    VLOG(6) << "sparse_weight_embedding API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = TensorToSelectedRows(weight);

    auto* kernel_out = SetKernelOutput(kernel_key.backend(), &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::EmbeddingInferMeta(MakeMetaTensor(*input_x),
                            MakeMetaTensor(*input_weight),
                            padding_idx,
                            sparse,
                            &meta_out);

    using kernel_signature = void (*)(const platform::DeviceContext&,
                                      const phi::DenseTensor&,
                                      const phi::SelectedRows&,
                                      int64_t,
                                      phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      (*kernel_fn)(*dev_ctx, *input_x, *input_weight, padding_idx, kernel_out);
    }
  }
  return api_output;
}

std::vector<Tensor> split_impl(const Tensor& x,
                               const IntArray& num_or_sections,
                               const Scalar& axis) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  Backend kernel_backend = kernel_key.backend();
  DataLayout kernel_layout = kernel_key.layout();
  DataType kernel_data_type = kernel_key.dtype();

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "split", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "split API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "split API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = PrepareData(x, kernel.InputAt(0), {});

  // Calculate the number of out tensors
  size_t out_number;
  if (num_or_sections.size() == 1) {
    out_number = num_or_sections.GetData()[0];
  } else {
    out_number = num_or_sections.size();
  }

  std::vector<Tensor> out;
  auto dense_outs = SetKernelOutput(out_number, kernel_backend, &out);
  std::vector<phi::MetaTensor> meta_outs;
  meta_outs.reserve(out_number);
  std::vector<phi::MetaTensor*> meta_out_ptrs;
  meta_out_ptrs.reserve(out_number);
  for (size_t i = 0; i < out_number; ++i) {
    meta_outs.push_back(dense_outs[i]);
    meta_out_ptrs.push_back(&meta_outs.back());
  }

  phi::SplitInferMeta(
      MakeMetaTensor(*dense_x), num_or_sections, axis, meta_out_ptrs);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::IntArray&,
                                    const phi::Scalar&,
                                    std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               phi::IntArray(num_or_sections),
               phi::Scalar(axis),
               dense_outs);

  return out;
}

std::tuple<Tensor, Tensor, Tensor> momentum_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& velocity,
    const Tensor& learning_rate,
    const paddle::optional<Tensor>& master_param,
    float mu,
    bool use_nesterov,
    const std::string& regularization_method,
    float regularization_coeff,
    bool multi_precision,
    float rescale_grad) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param);
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
  std::string kernel_name = "momentum";
  if (grad.is_selected_rows()) {
    kernel_name = "momentum_dense_param_sparse_grad";
  }
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << kernel_name << " API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << kernel_name << " API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_param = PrepareData(param, kernel.InputAt(0), {});
  auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
  auto input_velocity = PrepareData(velocity, kernel.InputAt(2), {});
  auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(3), {});
  auto input_master_param = PrepareData(master_param, kernel.InputAt(4), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = input_param.get();
  auto kernel_out_1 = input_velocity.get();
  phi::DenseTensor* kernel_out_2 = nullptr;
  if (input_master_param) {
    kernel_out_2 = input_master_param.get_ptr();
  }

  auto input_meta_ref_master_param = MakeMetaTensor(input_master_param);

  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  if (kernel_out_2) {
    phi::MetaTensor meta_out_2(kernel_out_2);
    phi::MomentumInferMeta(MakeMetaTensor(*input_param),
                           MakeMetaTensor(*input_grad),
                           MakeMetaTensor(*input_velocity),
                           MakeMetaTensor(*input_learning_rate),
                           input_meta_ref_master_param,
                           mu,
                           use_nesterov,
                           regularization_method,
                           regularization_coeff,
                           multi_precision,
                           rescale_grad,
                           &meta_out_0,
                           &meta_out_1,
                           &meta_out_2);
  } else {
    phi::MomentumInferMeta(MakeMetaTensor(*input_param),
                           MakeMetaTensor(*input_grad),
                           MakeMetaTensor(*input_velocity),
                           MakeMetaTensor(*input_learning_rate),
                           input_meta_ref_master_param,
                           mu,
                           use_nesterov,
                           regularization_method,
                           regularization_coeff,
                           multi_precision,
                           rescale_grad,
                           &meta_out_0,
                           &meta_out_1,
                           nullptr);
  }

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const paddle::optional<phi::DenseTensor>&,
                                    float,
                                    bool,
                                    const std::string&,
                                    float,
                                    bool,
                                    float,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  (*kernel_fn)(*dev_ctx,
               *input_param,
               *input_grad,
               *input_velocity,
               *input_learning_rate,
               input_master_param,
               mu,
               use_nesterov,
               regularization_method,
               regularization_coeff,
               multi_precision,
               rescale_grad,
               kernel_out_0,
               kernel_out_1,
               kernel_out_2);

  return api_output;
}

////////////////// Backward(grad) api impls //////////////////////

// TODO(chenweihang):  the original sum grad op can support higher-level
// differentiation,
// but if we use this impl, it will not support. We need to be able to reuse
// the autograd API here, which is not yet implemented
// TODO(chenweihang): we should support call generated api in custom api impl
void add_n_grad_impl(const std::vector<Tensor>& x,
                     const Tensor& out_grad,
                     std::vector<Tensor*> x_grad) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(out_grad);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  Backend kernel_backend = kernel_key.backend();
  DataLayout kernel_layout = kernel_key.layout();
  DataType kernel_data_type = kernel_key.dtype();

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "add_n_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "add_n_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_out_grad = PrepareData(out_grad, kernel.InputAt(0), {});

  auto dense_x_grad = SetKernelOutput(&x_grad);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::Scalar&,
                                    float,
                                    bool,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  for (auto* dense_x_grad_t : dense_x_grad) {
    phi::MetaTensor meta_out(dense_x_grad_t);
    phi::UnchangedInferMeta(MakeMetaTensor(*dense_out_grad), &meta_out);
    (*kernel_fn)(
        *dev_ctx, *dense_out_grad, phi::Scalar(1.0), 0.0, true, dense_x_grad_t);
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_impl(
    const Tensor& x,
    const Tensor& scale,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& variance,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    bool fuse_with_relu) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

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

  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "batch_norm", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "batch_norm API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "batch_norm API kernel: " << kernel << " asdf "
          << kernel.InputAt(0).layout << "asdfasdf " << kernel_layout;
  phi::TensorArgDef args0 = kernel.InputAt(0);
  phi::TensorArgDef args1 = kernel.InputAt(1);
  phi::TensorArgDef args2 = kernel.InputAt(2);
  phi::TensorArgDef args3 = kernel.InputAt(3);
  phi::TensorArgDef args4 = kernel.InputAt(4);
  if (kernel_backend == Backend::GPU) {
    args0.layout = x.layout();
    args1.layout = scale.layout();
    args2.layout = bias.layout();
    args3.layout = mean.layout();
    args4.layout = variance.layout();
  }

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_x = PrepareData(x, args0, {});
  auto input_scale = PrepareData(scale, args1, {});
  auto input_bias = PrepareData(bias, args2, {});
  auto input_mean = PrepareData(mean, args3, {});
  auto input_variance = PrepareData(variance, args4, {});

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  std::get<1>(api_output).set_impl(mean.impl());
  std::get<2>(api_output).set_impl(variance.impl());
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
  auto kernel_out_4 = SetKernelOutput(kernel_backend, &std::get<4>(api_output));
  auto kernel_out_5 = SetKernelOutput(kernel_backend, &std::get<5>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);
  phi::MetaTensor meta_out_4(kernel_out_4);
  phi::MetaTensor meta_out_5(kernel_out_5);

  phi::BatchNormInferMeta(MakeMetaTensor(*input_x),
                          MakeMetaTensor(*input_scale),
                          MakeMetaTensor(*input_bias),
                          MakeMetaTensor(*input_mean),
                          MakeMetaTensor(*input_variance),
                          momentum,
                          epsilon,
                          data_layout,
                          is_test,
                          use_global_stats,
                          trainable_statistics,
                          fuse_with_relu,
                          &meta_out_0,
                          &meta_out_1,
                          &meta_out_2,
                          &meta_out_3,
                          &meta_out_4,
                          &meta_out_5);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    float,
                                    float,
                                    const std::string&,
                                    bool,
                                    bool,
                                    bool,
                                    bool,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    (*kernel_fn)(*dev_ctx,
                 *input_x,
                 *input_scale,
                 *input_bias,
                 *input_mean,
                 *input_variance,
                 momentum,
                 epsilon,
                 data_layout,
                 is_test,
                 use_global_stats,
                 trainable_statistics,
                 fuse_with_relu,
                 kernel_out_0,
                 kernel_out_1,
                 kernel_out_2,
                 kernel_out_3,
                 kernel_out_4,
                 kernel_out_5);
  }

  return api_output;
}

void imag_grad_impl(const Tensor& out_grad, Tensor* x_grad) {
  phi::KernelKey kernel_key{ParseBackend(out_grad),
                            out_grad.layout(),
                            phi::dtype::ToComplex(out_grad.dtype())};
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "imag_grad", kernel_key);

  VLOG(6) << "imag_grad API kernel key: " << kernel_key;
  VLOG(6) << "imag_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_out_grad = TensorToDenseTensor(out_grad);

  auto kernel_out = SetKernelOutput(kernel_key.backend(), x_grad);
  phi::MetaTensor meta_out(kernel_out);
  phi::RealAndImagGradInferMeta(*dense_out_grad, &meta_out);

  using kernel_signature = void (*)(
      const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);

  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *dense_out_grad, kernel_out);
}

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

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  if (phi::DenseTensor::classof(weight.impl().get())) {
    std::string kernel_name =
        sparse ? "embedding_sparse_grad" : "embedding_grad";
    const auto& kernel =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = PrepareData(weight, kernel.InputAt(1), {});
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out =
          SetSelectedRowsKernelOutput(kernel_key.backend(), weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      meta_out.set_dims(input_weight->dims());
      meta_out.set_dtype(input_weight->dtype());
      kernel_out->set_height(input_weight->dims()[0]);

      using kernel_signature = void (*)(const platform::DeviceContext&,
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
      auto* kernel_out = SetKernelOutput(kernel_key.backend(), weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      phi::UnchangedInferMeta(MakeMetaTensor(*input_weight), &meta_out);
      using kernel_signature = void (*)(const platform::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const phi::DenseTensor&,
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
  } else {
    std::string kernel_name = sparse ? "sparse_weight_embedding_sparse_grad"
                                     : "sparse_weight_embedding_grad";
    const auto& kernel =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = TensorToSelectedRows(weight);
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out =
          SetSelectedRowsKernelOutput(kernel_key.backend(), weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      phi::UnchangedInferMeta(MakeMetaTensor(*input_weight), &meta_out);
      using kernel_signature = void (*)(const platform::DeviceContext&,
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
      auto* kernel_out = SetKernelOutput(kernel_key.backend(), weight_grad);
      phi::MetaTensor meta_out(kernel_out);
      meta_out.set_dims(input_weight->GetCompleteDims());
      meta_out.set_dtype(input_weight->dtype());
      using kernel_signature = void (*)(const platform::DeviceContext&,
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

void real_grad_impl(const Tensor& out_grad, Tensor* x_grad) {
  phi::KernelKey kernel_key{ParseBackend(out_grad),
                            out_grad.layout(),
                            phi::dtype::ToComplex(out_grad.dtype())};
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "real_grad", kernel_key);

  VLOG(6) << "real_grad API kernel key: " << kernel_key;
  VLOG(6) << "real_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_out_grad = TensorToDenseTensor(out_grad);

  auto kernel_out = SetKernelOutput(kernel_key.backend(), x_grad);
  phi::MetaTensor meta_out(kernel_out);
  phi::RealAndImagGradInferMeta(*dense_out_grad, &meta_out);

  using kernel_signature = void (*)(
      const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);

  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *dense_out_grad, kernel_out);
}

}  // namespace experimental
}  // namespace paddle
