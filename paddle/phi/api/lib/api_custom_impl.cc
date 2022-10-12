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
        void (*)(const platform::DeviceContext&,
                 const std::vector<const phi::SelectedRows*>&,
                 phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  } else {
    std::vector<const phi::TensorBase*> input_x(x.size());
    for (size_t i = 0; i < input_x.size(); ++i) {
      input_x[i] = x[i].impl().get();
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
        void (*)(const platform::DeviceContext&,
                 const std::vector<const phi::TensorBase*>&,
                 phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  }

  return api_output;
}

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking) {
  Tensor out;
  copy(x, place, blocking, &out);
  return out;
}

////////////////// Backward(grad) api impls //////////////////////

void imag_grad_impl(const Tensor& out_grad, Tensor* x_grad) {
  phi::KernelKey kernel_key{ParseBackend(out_grad),
                            out_grad.layout(),
                            phi::dtype::ToComplex(out_grad.dtype())};
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "imag_grad", kernel_key);
  const auto& kernel = kernel_result.kernel;

  VLOG(6) << "imag_grad API kernel key: " << kernel_key;
  VLOG(6) << "imag_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_out_grad = TensorToDenseTensor(out_grad);

  auto kernel_out = SetKernelOutput(x_grad);
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
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = PrepareData(weight, kernel.InputAt(1), {});
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out = SetSelectedRowsKernelOutput(weight_grad);
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
      auto* kernel_out = SetKernelOutput(weight_grad);
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
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            kernel_name,
            {kernel_key.backend(), kernel_key.layout(), kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << kernel_name << " API kernel: " << kernel;

    auto input_x = PrepareData(x, kernel.InputAt(0), {});
    auto input_weight = TensorToSelectedRows(weight);
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

    if (sparse) {
      auto* kernel_out = SetSelectedRowsKernelOutput(weight_grad);
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
      auto* kernel_out = SetKernelOutput(weight_grad);
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
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "real_grad", kernel_key);
  const auto& kernel = kernel_result.kernel;

  VLOG(6) << "real_grad API kernel key: " << kernel_key;
  VLOG(6) << "real_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_out_grad = TensorToDenseTensor(out_grad);

  auto kernel_out = SetKernelOutput(x_grad);
  phi::MetaTensor meta_out(kernel_out);
  phi::RealAndImagGradInferMeta(*dense_out_grad, &meta_out);

  using kernel_signature = void (*)(
      const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);

  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *dense_out_grad, kernel_out);
}

}  // namespace experimental
}  // namespace paddle
