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
#include "paddle/common/flags.h"
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
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle::experimental {

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
    if (phi::DenseTensor::classof(input.impl().get()) ||
        phi::distributed::DistTensor::classof(input.impl().get())) {
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
#ifdef PADDLE_WITH_DISTRIBUTE
    bool run_auto_parallel = AllInputsAreDistTensor(x);
    bool rank_is_in_current_mesh = true;
    if (run_auto_parallel) {
      auto mesh =
          std::static_pointer_cast<phi::distributed::DistTensor>(x[0].impl())
              ->dist_attr()
              .process_mesh();
      rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);

      std::vector<const phi::TensorBase*> input_x(x.size());
      for (size_t i = 0; i < input_x.size(); ++i) {
        input_x[i] = x[i].impl().get();
      }

      auto meta_dist_input_x = MakeDistMetaTensor(input_x);
      auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(
          meta_dist_input_x);

      auto dist_out = SetKernelDistOutput(&api_output);
      auto dense_out = dist_out->unsafe_mutable_value();
      if (!rank_is_in_current_mesh) {
        *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(
                nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
      }

      phi::MetaTensor meta_dist_out(dist_out);
      auto x_meta_vec = MakeMetaTensor(input_x);
      std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
      for (size_t i = 0; i < x_meta_vec.size(); ++i) {
        x_metas[i] = &x_meta_vec[i];
      }
      phi::AddNInferMeta(x_metas, &meta_dist_out);
      if (rank_is_in_current_mesh) {
        auto dist_input_x =
            ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0]);
        dist_input_x = PrepareDataForDistTensor(
            dist_input_x,
            GetKernelInputArgDef(kernel.InputAt(0), kernel_backend),
            {},
            kernel_result.is_stride_kernel);
        std::vector<const phi::TensorBase*> input_x(dist_input_x.size());
        for (size_t i = 0; i < dist_input_x.size(); ++i) {
          input_x[i] = dist_input_x[i]->unsafe_mutable_value();
        }

        auto x_meta_vec = MakeMetaTensor(input_x);
        std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
        for (size_t i = 0; i < x_meta_vec.size(); ++i) {
          x_metas[i] = &x_meta_vec[i];
        }
        phi::MetaTensor meta_dense_out(dense_out);
        phi::AddNInferMeta(x_metas, &meta_dense_out);

        using kernel_signature =
            void (*)(const phi::DeviceContext&,
                     const std::vector<const phi::TensorBase*>&,
                     phi::DenseTensor*);
        auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
        (*kernel_fn)(*dev_ctx, input_x, dense_out);
      }
      PADDLE_ENFORCE_EQ(paddle::holds_alternative<
                            std::vector<phi::distributed::TensorDistAttr>>(
                            spmd_info.first[0]),
                        true,
                        common::errors::PreconditionNotMet(
                            "Arg must be a vector of TensorDistAttr"));

      auto current_process_mesh =
          paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
      SetReplicatedDistAttrForOutput(dist_out, current_process_mesh);
      return api_output;
    }
#endif
    std::vector<const phi::TensorBase*> input_x(x.size());
    std::vector<std::shared_ptr<phi::DenseTensor>> temp_dense_tensors;
    temp_dense_tensors.reserve(x.size());
    for (size_t i = 0; i < input_x.size(); ++i) {
      if (phi::DenseTensor::classof(x[i].impl().get())) {
        temp_dense_tensors.push_back(
            PrepareData(x[i], kernel.InputAt(0), {}, false));
        input_x[i] = temp_dense_tensors.back().get();
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

    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
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

std::tuple<Tensor, Tensor> fused_gemm_epilogue_impl(
    const Tensor& x,
    const Tensor& y,
    const Tensor& bias,
    bool trans_x,
    bool trans_y,
    const std::string& activation) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, bias);
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

  VLOG(6) << "fused_gemm_epilogue API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_gemm_epilogue",
      {kernel_backend, kernel_layout, kernel_data_type},
      true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList(
        "fused_gemm_epilogue", kernel_data_type);
  }
  VLOG(6) << "fused_gemm_epilogue kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential
  // falling-back to CPU
  Backend actual_kernel_backend =
      kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(
      x,
      GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);
  auto input_y = PrepareData(
      y,
      GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(
      bias,
      GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  phi::DenseTensor* kernel_out_1 = nullptr;
  if (activation != "none") {
    kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  }

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FusedGemmEpilogueInferMeta(MakeMetaTensor(*input_x),
                                  MakeMetaTensor(*input_y),
                                  MakeMetaTensor(*input_bias),
                                  trans_x,
                                  trans_y,
                                  activation,
                                  kernel_out_0 ? &meta_out_0 : nullptr,
                                  kernel_out_1 ? &meta_out_1 : nullptr);

  using kernel_signature = void (*)(const phi::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    bool,
                                    bool,
                                    const std::string&,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  (*kernel_fn)(*dev_ctx,
               *input_x,
               *input_y,
               *input_bias,
               trans_x,
               trans_y,
               activation,
               kernel_out_0,
               kernel_out_1);

  if (kernel_result.has_fallback_cpu) {
    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }
  return api_output;
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

  if (phi::DenseTensor::classof(weight.impl().get()) ||
      phi::distributed::DistTensor::classof(weight.impl().get())) {
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

#ifdef PADDLE_WITH_DISTRIBUTE
    bool run_auto_parallel = AllInputsAreDistTensor(x, weight, out_grad);
    // Auto Parallel condition
    if (run_auto_parallel) {
      bool rank_is_in_current_mesh = true;
      auto mesh =
          std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())
              ->dist_attr()
              .process_mesh();
      rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);

      // 1. InferSpmd (Infer DistAttr of Inputs&Outputs)
      auto meta_dist_input_x = MakeDistMetaTensor(*x.impl());
      auto meta_dist_input_weight = MakeDistMetaTensor(*weight.impl());
      auto meta_dist_input_out_grad = MakeDistMetaTensor(*out_grad.impl());
      auto spmd_info =
          phi::distributed::EmbeddingGradInferSpmd(meta_dist_input_x,
                                                   meta_dist_input_weight,
                                                   meta_dist_input_out_grad,
                                                   padding_idx,
                                                   sparse);

      // 2. Create Temporary Output & Prepare Dist and Dense Output
      std::shared_ptr<phi::distributed::DistTensor> shared_dist_out =
          CreateKernelDistOutput(
              weight_grad, !rank_is_in_current_mesh, spmd_info.second[0]);
      phi::distributed::DistTensor* dist_out = shared_dist_out.get();
      phi::DenseTensor* dense_out = dist_out->unsafe_mutable_value();
      if (dense_out && !rank_is_in_current_mesh && !dist_out->defined()) {
        *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(
                nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
      }

      // 3. Infer DistTensor's Global Shape
      phi::MetaTensor meta_dist_out(dist_out);
      UnchangedInferMeta(MakeMetaTensor(*weight.impl()), &meta_dist_out);

      // 4. Set Output Dist Attr For Default Impl

      if (rank_is_in_current_mesh) {
        // 5. Reshard Input
        auto dist_input_x =
            ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0]);
        auto dist_input_weight =
            ReshardApiInputToKernelInput(dev_ctx, weight, spmd_info.first[1]);
        auto dist_input_out_grad =
            ReshardApiInputToKernelInput(dev_ctx, out_grad, spmd_info.first[2]);

        // 6. PrepareData (DataTransform & Prepare Dense Input)
        dist_input_x = PrepareDataForDistTensor(
            dist_input_x,
            GetKernelInputArgDef(kernel.InputAt(0), kernel_key.backend()),
            {},
            kernel_result.is_stride_kernel);
        auto input_x = &dist_input_x->value();
        dist_input_weight = PrepareDataForDistTensor(
            dist_input_weight,
            GetKernelInputArgDef(kernel.InputAt(1), kernel_key.backend()),
            {},
            kernel_result.is_stride_kernel);
        auto input_weight = &dist_input_weight->value();
        dist_input_out_grad = PrepareDataForDistTensor(
            dist_input_out_grad,
            GetKernelInputArgDef(kernel.InputAt(2), kernel_key.backend()),
            {},
            kernel_result.is_stride_kernel);
        auto input_out_grad = &dist_input_out_grad->value();

        // 7. Infer Local DenseTensor Meta
        phi::MetaTensor meta_dense_out(dense_out);
        phi::EmbeddingGradInferMeta(MakeMetaTensor(*input_x),
                                    MakeMetaTensor(*input_weight),
                                    &meta_dense_out);

        // 8. DenseTensor Kernel Call
        using kernel_signature = void (*)(const phi::DeviceContext&,
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
                     dense_out);
      }
      // 9. Reshard Kernel Output to API output
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out, weight_grad);

      // 10. Return
      return;
    }
#endif  // PADDLE_WITH_DISTRIBUTE

    auto input_x = PrepareData(x, kernel.InputAt(0), {}, false);
    auto input_weight = PrepareData(weight, kernel.InputAt(1), {}, false);
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {}, false);

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

    auto* dev_ctx = GetDeviceContextByBackend(
        kernel_result.has_fallback_cpu ? Backend::CPU : kernel_key.backend());

    auto input_x = PrepareData(x, kernel.InputAt(0), {}, false);
    auto input_weight = TensorToSelectedRows(weight);
    auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {}, false);

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

}  // namespace paddle::experimental
