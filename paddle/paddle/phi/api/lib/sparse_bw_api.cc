
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/backward.h"

#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/backward.h"

COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle {
namespace experimental {
namespace sparse {


PADDLE_API void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "abs_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("abs_grad", kernel_data_type);
    }
    VLOG(6) << "abs_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "abs_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("abs_grad", kernel_data_type);
    }
    VLOG(6) << "abs_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (abs_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void acos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "acos_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acos_grad", kernel_data_type);
    }
    VLOG(6) << "acos_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "acos_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acos_grad", kernel_data_type);
    }
    VLOG(6) << "acos_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acos_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void acosh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "acosh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acosh_grad", kernel_data_type);
    }
    VLOG(6) << "acosh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "acosh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acosh_grad", kernel_data_type);
    }
    VLOG(6) << "acosh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acosh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "add_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add_grad", kernel_data_type);
    }
    VLOG(6) << "add_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "add_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add_grad", kernel_data_type);
    }
    VLOG(6) << "add_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get()) && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "add_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add_grad", kernel_data_type);
    }
    VLOG(6) << "add_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (add_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void addmm_grad(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, Tensor* input_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, x, y, out_grad);
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

  if (phi::DenseTensor::classof(input.impl().get()) && x.is_sparse_csr_tensor() && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm_grad", kernel_data_type);
    }
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_input = PrepareDataForDenseTensorInSparse(input);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (input.is_sparse_csr_tensor() && x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm_grad", kernel_data_type);
    }
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_input = PrepareDataForSparseCsrTensor(input);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (phi::DenseTensor::classof(input.impl().get()) && x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm_grad", kernel_data_type);
    }
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_input = PrepareDataForDenseTensorInSparse(input);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (input.is_sparse_coo_tensor() && x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm_grad", kernel_data_type);
    }
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_input = PrepareDataForSparseCooTensor(input);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (addmm_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void asin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "asin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asin_grad", kernel_data_type);
    }
    VLOG(6) << "asin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "asin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asin_grad", kernel_data_type);
    }
    VLOG(6) << "asin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asin_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void asinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "asinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asinh_grad", kernel_data_type);
    }
    VLOG(6) << "asinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "asinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asinh_grad", kernel_data_type);
    }
    VLOG(6) << "asinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asinh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void atan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "atan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atan_grad", kernel_data_type);
    }
    VLOG(6) << "atan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "atan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atan_grad", kernel_data_type);
    }
    VLOG(6) << "atan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atan_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void atanh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "atanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atanh_grad", kernel_data_type);
    }
    VLOG(6) << "atanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "atanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atanh_grad", kernel_data_type);
    }
    VLOG(6) << "atanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atanh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias, mean_out, variance_out, saved_mean, saved_variance, reserve_space, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(scale.impl().get()) && phi::DenseTensor::classof(bias.impl().get()) && (!mean_out || phi::DenseTensor::classof(mean_out->impl().get())) && (!variance_out || phi::DenseTensor::classof(variance_out->impl().get())) && phi::DenseTensor::classof(saved_mean.impl().get()) && phi::DenseTensor::classof(saved_variance.impl().get()) && (!reserve_space || phi::DenseTensor::classof(reserve_space->impl().get())) && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "batch_norm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "batch_norm_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("batch_norm_grad", kernel_data_type);
    }
    VLOG(6) << "batch_norm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(scale_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_2 = SetSparseKernelOutput(bias_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_scale = PrepareDataForDenseTensorInSparse(scale);
    auto input_bias = PrepareDataForDenseTensorInSparse(bias);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_scale), MakeMetaTensor(*input_bias), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_mean_out = PrepareDataForDenseTensorInSparse(mean_out);
    auto input_variance_out = PrepareDataForDenseTensorInSparse(variance_out);
    auto input_saved_mean = PrepareDataForDenseTensorInSparse(saved_mean);
    auto input_saved_variance = PrepareDataForDenseTensorInSparse(saved_variance);
    auto input_reserve_space = PrepareDataForDenseTensorInSparse(reserve_space);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_scale.get());
    kernel_context.EmplaceBackInput(input_bias.get());
    kernel_context.EmplaceBackInput(mean_out ? &(*input_mean_out) : nullptr);
    kernel_context.EmplaceBackInput(variance_out ? &(*input_variance_out) : nullptr);
    kernel_context.EmplaceBackInput(input_saved_mean.get());
    kernel_context.EmplaceBackInput(input_saved_variance.get());
    kernel_context.EmplaceBackInput(reserve_space ? &(*input_reserve_space) : nullptr);
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(momentum);
    kernel_context.EmplaceBackAttr(epsilon);
    kernel_context.EmplaceBackAttr(data_format);
    kernel_context.EmplaceBackAttr(is_test);
    kernel_context.EmplaceBackAttr(use_global_stats);
    kernel_context.EmplaceBackAttr(trainable_statistics);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (batch_norm_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void cast_grad(const Tensor& x, const Tensor& out_grad, DataType value_dtype, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "cast_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("cast_grad", kernel_data_type);
    }
    VLOG(6) << "cast_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "cast_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("cast_grad", kernel_data_type);
    }
    VLOG(6) << "cast_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (cast_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void conv3d_grad(const Tensor& x, const Tensor& kernel, const Tensor& out, const Tensor& rulebook, const Tensor& counter, const Tensor& out_grad, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key, Tensor* x_grad, Tensor* kernel_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, kernel, out, rulebook, counter, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(kernel.impl().get()) && out.is_sparse_coo_tensor() && phi::DenseTensor::classof(rulebook.impl().get()) && phi::DenseTensor::classof(counter.impl().get()) && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "conv3d_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "conv3d_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("conv3d_grad", kernel_data_type);
    }
    VLOG(6) << "conv3d_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(kernel_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_kernel = PrepareDataForDenseTensorInSparse(kernel);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_kernel), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out = PrepareDataForSparseCooTensor(out);
    auto input_rulebook = PrepareDataForDenseTensorInSparse(rulebook);
    auto input_counter = PrepareDataForDenseTensorInSparse(counter);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_kernel.get());
    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_rulebook.get());
    kernel_context.EmplaceBackInput(input_counter.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackAttr(groups);
    kernel_context.EmplaceBackAttr(subm);
    kernel_context.EmplaceBackAttr(key);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out, out_grad);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "divide_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide_grad", kernel_data_type);
    }
    VLOG(6) << "divide_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out = PrepareDataForSparseCooTensor(out);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "divide_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide_grad", kernel_data_type);
    }
    VLOG(6) << "divide_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out = PrepareDataForSparseCsrTensor(out);
    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void expm1_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "expm1_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("expm1_grad", kernel_data_type);
    }
    VLOG(6) << "expm1_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "expm1_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("expm1_grad", kernel_data_type);
    }
    VLOG(6) << "expm1_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (expm1_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void leaky_relu_grad(const Tensor& x, const Tensor& out_grad, float alpha, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "leaky_relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("leaky_relu_grad", kernel_data_type);
    }
    VLOG(6) << "leaky_relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "leaky_relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("leaky_relu_grad", kernel_data_type);
    }
    VLOG(6) << "leaky_relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (leaky_relu_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void log1p_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "log1p_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("log1p_grad", kernel_data_type);
    }
    VLOG(6) << "log1p_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "log1p_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("log1p_grad", kernel_data_type);
    }
    VLOG(6) << "log1p_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (log1p_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void masked_matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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

  if (phi::DenseTensor::classof(x.impl().get()) && phi::DenseTensor::classof(y.impl().get()) && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "masked_matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "masked_matmul_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("masked_matmul_grad", kernel_data_type);
    }
    VLOG(6) << "masked_matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForDenseTensorInSparse(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (masked_matmul_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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

  if (x.is_sparse_csr_tensor() && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul_grad", kernel_data_type);
    }
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul_grad", kernel_data_type);
    }
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul_grad", kernel_data_type);
    }
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul_grad", kernel_data_type);
    }
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (matmul_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void maxpool_grad(const Tensor& x, const Tensor& rulebook, const Tensor& counter, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_sizes, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, rulebook, counter, out, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(rulebook.impl().get()) && phi::DenseTensor::classof(counter.impl().get()) && out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "maxpool_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "maxpool_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("maxpool_grad", kernel_data_type);
    }
    VLOG(6) << "maxpool_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_rulebook = PrepareDataForDenseTensorInSparse(rulebook);
    auto input_counter = PrepareDataForDenseTensorInSparse(counter);
    auto input_out = PrepareDataForSparseCooTensor(out);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_rulebook.get());
    kernel_context.EmplaceBackInput(input_counter.get());
    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(kernel_sizes);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "multiply_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("multiply_grad", kernel_data_type);
    }
    VLOG(6) << "multiply_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "multiply_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("multiply_grad", kernel_data_type);
    }
    VLOG(6) << "multiply_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (multiply_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void mv_grad(const Tensor& x, const Tensor& vec, const Tensor& out_grad, Tensor* x_grad, Tensor* vec_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, vec, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(vec.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "mv_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("mv_grad", kernel_data_type);
    }
    VLOG(6) << "mv_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(vec_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_vec = PrepareDataForDenseTensorInSparse(vec);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_vec), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_vec.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && phi::DenseTensor::classof(vec.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "mv_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("mv_grad", kernel_data_type);
    }
    VLOG(6) << "mv_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(vec_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_vec = PrepareDataForDenseTensorInSparse(vec);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_vec), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_vec.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (mv_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void pow_grad(const Tensor& x, const Tensor& out_grad, float factor, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "pow_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("pow_grad", kernel_data_type);
    }
    VLOG(6) << "pow_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "pow_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("pow_grad", kernel_data_type);
    }
    VLOG(6) << "pow_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (pow_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void relu6_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "relu6_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu6_grad", kernel_data_type);
    }
    VLOG(6) << "relu6_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "relu6_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu6_grad", kernel_data_type);
    }
    VLOG(6) << "relu6_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu6_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu_grad", kernel_data_type);
    }
    VLOG(6) << "relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu_grad", kernel_data_type);
    }
    VLOG(6) << "relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void reshape_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "reshape_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "reshape_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape_grad", kernel_data_type);
    }
    VLOG(6) << "reshape_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "reshape_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "reshape_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape_grad", kernel_data_type);
    }
    VLOG(6) << "reshape_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (reshape_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sin_grad", kernel_data_type);
    }
    VLOG(6) << "sin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "sin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sin_grad", kernel_data_type);
    }
    VLOG(6) << "sin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sin_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sinh_grad", kernel_data_type);
    }
    VLOG(6) << "sinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "sinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sinh_grad", kernel_data_type);
    }
    VLOG(6) << "sinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sinh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "softmax_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("softmax_grad", kernel_data_type);
    }
    VLOG(6) << "softmax_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "softmax_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("softmax_grad", kernel_data_type);
    }
    VLOG(6) << "softmax_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (softmax_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sparse_coo_tensor_grad(const Tensor& indices, const Tensor& out_grad, Tensor* values_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(indices, out_grad);
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

  if (phi::DenseTensor::classof(indices.impl().get()) && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sparse_coo_tensor_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sparse_coo_tensor_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sparse_coo_tensor_grad", kernel_data_type);
    }
    VLOG(6) << "sparse_coo_tensor_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(values_grad, TensorType::DENSE_TENSOR);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out_grad), &meta_out);

    auto input_indices = PrepareDataForDenseTensorInSparse(indices);

    kernel_context.EmplaceBackInput(input_indices.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sparse_coo_tensor_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sqrt_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sqrt_grad", kernel_data_type);
    }
    VLOG(6) << "sqrt_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "sqrt_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sqrt_grad", kernel_data_type);
    }
    VLOG(6) << "sqrt_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sqrt_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void square_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "square_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("square_grad", kernel_data_type);
    }
    VLOG(6) << "square_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "square_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("square_grad", kernel_data_type);
    }
    VLOG(6) << "square_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (square_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "subtract_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("subtract_grad", kernel_data_type);
    }
    VLOG(6) << "subtract_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "subtract_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("subtract_grad", kernel_data_type);
    }
    VLOG(6) << "subtract_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (subtract_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sum_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sum_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sum_grad", kernel_data_type);
    }
    VLOG(6) << "sum_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axis));
    kernel_context.EmplaceBackAttr(keepdim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "sum_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sum_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sum_grad", kernel_data_type);
    }
    VLOG(6) << "sum_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axis));
    kernel_context.EmplaceBackAttr(keepdim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sum_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sync_batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias, saved_mean, saved_variance, reserve_space, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(scale.impl().get()) && phi::DenseTensor::classof(bias.impl().get()) && phi::DenseTensor::classof(saved_mean.impl().get()) && phi::DenseTensor::classof(saved_variance.impl().get()) && (!reserve_space || phi::DenseTensor::classof(reserve_space->impl().get())) && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "sync_batch_norm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sync_batch_norm_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sync_batch_norm_grad", kernel_data_type);
    }
    VLOG(6) << "sync_batch_norm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(scale_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_2 = SetSparseKernelOutput(bias_grad, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_scale = PrepareDataForDenseTensorInSparse(scale);
    auto input_bias = PrepareDataForDenseTensorInSparse(bias);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GeneralTernaryGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_scale), MakeMetaTensor(*input_bias), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

    auto input_saved_mean = PrepareDataForDenseTensorInSparse(saved_mean);
    auto input_saved_variance = PrepareDataForDenseTensorInSparse(saved_variance);
    auto input_reserve_space = PrepareDataForDenseTensorInSparse(reserve_space);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_scale.get());
    kernel_context.EmplaceBackInput(input_bias.get());
    kernel_context.EmplaceBackInput(input_saved_mean.get());
    kernel_context.EmplaceBackInput(input_saved_variance.get());
    kernel_context.EmplaceBackInput(reserve_space ? &(*input_reserve_space) : nullptr);
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(momentum);
    kernel_context.EmplaceBackAttr(epsilon);
    kernel_context.EmplaceBackAttr(data_format);
    kernel_context.EmplaceBackAttr(is_test);
    kernel_context.EmplaceBackAttr(use_global_stats);
    kernel_context.EmplaceBackAttr(trainable_statistics);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sync_batch_norm_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void tan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "tan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tan_grad", kernel_data_type);
    }
    VLOG(6) << "tan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "tan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tan_grad", kernel_data_type);
    }
    VLOG(6) << "tan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tan_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void tanh_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
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

  if (out.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "tanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tanh_grad", kernel_data_type);
    }
    VLOG(6) << "tanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out = PrepareDataForSparseCooTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "tanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tanh_grad", kernel_data_type);
    }
    VLOG(6) << "tanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out = PrepareDataForSparseCsrTensor(out);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_out.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tanh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void to_dense_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "to_dense_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_to_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_dense_grad", kernel_data_type);
    }
    VLOG(6) << "to_dense_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (to_dense_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void to_sparse_coo_grad(const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out_grad);
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

  if (out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "to_sparse_coo_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_to_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_sparse_coo_grad", kernel_data_type);
    }
    VLOG(6) << "to_sparse_coo_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::DENSE_TENSOR);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_out_grad), &meta_out);


    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (to_sparse_coo_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void transpose_grad(const Tensor& out_grad, const std::vector<int>& perm, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out_grad);
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

  if (out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "transpose_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "transpose_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("transpose_grad", kernel_data_type);
    }
    VLOG(6) << "transpose_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::TransposeGradInferMeta(MakeMetaTensor(*input_out_grad), perm, &meta_out);


    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(perm);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "transpose_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "transpose_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("transpose_grad", kernel_data_type);
    }
    VLOG(6) << "transpose_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::TransposeGradInferMeta(MakeMetaTensor(*input_out_grad), perm, &meta_out);


    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(perm);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (transpose_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void values_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "values_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "values_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("values_grad", kernel_data_type);
    }
    VLOG(6) << "values_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (values_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void fused_attention_grad(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& softmax, const Tensor& out_grad, Tensor* query_grad, Tensor* key_grad, Tensor* value_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(softmax);

  kernel_data_type = ParseDataType(query);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, softmax, out_grad);
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

  if (phi::DenseTensor::classof(query.impl().get()) && phi::DenseTensor::classof(key.impl().get()) && phi::DenseTensor::classof(value.impl().get()) && softmax.is_sparse_csr_tensor() && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "fused_attention_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "fused_attention_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_attention_grad", kernel_data_type);
    }
    VLOG(6) << "fused_attention_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(query_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(key_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_2 = SetSparseKernelOutput(value_grad, TensorType::DENSE_TENSOR);
    auto input_query = PrepareDataForDenseTensorInSparse(query);
    auto input_key = PrepareDataForDenseTensorInSparse(key);
    auto input_value = PrepareDataForDenseTensorInSparse(value);
    auto input_softmax = PrepareDataForSparseCsrTensor(softmax);
    auto input_out_grad = PrepareDataForDenseTensorInSparse(out_grad);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::sparse::FusedAttentionGradInferMeta(MakeMetaTensor(*input_query), MakeMetaTensor(*input_key), MakeMetaTensor(*input_value), MakeMetaTensor(*input_softmax), MakeMetaTensor(*input_out_grad), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


    kernel_context.EmplaceBackInput(input_query.get());
    kernel_context.EmplaceBackInput(input_key.get());
    kernel_context.EmplaceBackInput(input_value.get());
    kernel_context.EmplaceBackInput(input_softmax.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void slice_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axes, const IntArray& starts, const IntArray& ends, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
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

  if (x.is_sparse_coo_tensor() && out_grad.is_sparse_coo_tensor()) {

    VLOG(6) << "slice_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "slice_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("slice_grad", kernel_data_type);
    }
    VLOG(6) << "slice_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCooTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axes));
    kernel_context.EmplaceBackAttr(phi::IntArray(starts));
    kernel_context.EmplaceBackAttr(phi::IntArray(ends));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.is_sparse_csr_tensor() && out_grad.is_sparse_csr_tensor()) {

    VLOG(6) << "slice_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "slice_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("slice_grad", kernel_data_type);
    }
    VLOG(6) << "slice_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    auto input_out_grad = PrepareDataForSparseCsrTensor(out_grad);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_out_grad.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axes));
    kernel_context.EmplaceBackAttr(phi::IntArray(starts));
    kernel_context.EmplaceBackAttr(phi::IntArray(ends));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (slice_grad) for input tensors is unimplemented, please check the type of input tensors."));
}


}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
