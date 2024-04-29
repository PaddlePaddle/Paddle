
#include "paddle/phi/api/include/sparse_api.h"
#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/utils/none.h"

#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/multiary.h"

COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle {
namespace experimental {
namespace sparse {


PADDLE_API Tensor abs(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "abs api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("abs", kernel_data_type);
    }
    VLOG(6) << "abs api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "abs api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("abs", kernel_data_type);
    }
    VLOG(6) << "abs api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (abs) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor acos(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "acos api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acos", kernel_data_type);
    }
    VLOG(6) << "acos api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "acos api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acos", kernel_data_type);
    }
    VLOG(6) << "acos api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acos) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor acosh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "acosh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acosh", kernel_data_type);
    }
    VLOG(6) << "acosh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "acosh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("acosh", kernel_data_type);
    }
    VLOG(6) << "acosh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acosh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor add(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "add api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add", kernel_data_type);
    }
    VLOG(6) << "add api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "add api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add", kernel_data_type);
    }
    VLOG(6) << "add api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "add api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("add", kernel_data_type);
    }
    VLOG(6) << "add api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (add) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor asin(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "asin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asin", kernel_data_type);
    }
    VLOG(6) << "asin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "asin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asin", kernel_data_type);
    }
    VLOG(6) << "asin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asin) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor asinh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "asinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asinh", kernel_data_type);
    }
    VLOG(6) << "asinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "asinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("asinh", kernel_data_type);
    }
    VLOG(6) << "asinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asinh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor atan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "atan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atan", kernel_data_type);
    }
    VLOG(6) << "atan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "atan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atan", kernel_data_type);
    }
    VLOG(6) << "atan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor atanh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "atanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atanh", kernel_data_type);
    }
    VLOG(6) << "atanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "atanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("atanh", kernel_data_type);
    }
    VLOG(6) << "atanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atanh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, mean, variance, scale, bias);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(mean.impl().get()) && phi::DenseTensor::classof(variance.impl().get()) && phi::DenseTensor::classof(scale.impl().get()) && phi::DenseTensor::classof(bias.impl().get())) {

    VLOG(6) << "batch_norm_ api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "batch_norm_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("batch_norm_", kernel_data_type);
    }
    VLOG(6) << "batch_norm_ api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> api_output{Tensor(), mean, variance, Tensor(), Tensor(), Tensor()};
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_3 = SetSparseKernelOutput(&std::get<3>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_4 = SetSparseKernelOutput(&std::get<4>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_5 = SetSparseKernelOutput(&std::get<5>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_mean = PrepareDataForDenseTensorInSparse(mean);
    auto input_variance = PrepareDataForDenseTensorInSparse(variance);
    auto input_scale = PrepareDataForDenseTensorInSparse(scale);
    auto input_bias = PrepareDataForDenseTensorInSparse(bias);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_5(kernel_out_5, kernel_result.is_stride_kernel);

  phi::BatchNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_mean), MakeMetaTensor(*input_variance), MakeMetaTensor(*input_scale), MakeMetaTensor(*input_bias), is_test, momentum, epsilon, data_format, use_global_stats, trainable_statistics, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_mean.get());
    kernel_context.EmplaceBackInput(input_variance.get());
    kernel_context.EmplaceBackInput(input_scale.get());
    kernel_context.EmplaceBackInput(input_bias.get());
    kernel_context.EmplaceBackAttr(is_test);
    kernel_context.EmplaceBackAttr(momentum);
    kernel_context.EmplaceBackAttr(epsilon);
    kernel_context.EmplaceBackAttr(data_format);
    kernel_context.EmplaceBackAttr(use_global_stats);
    kernel_context.EmplaceBackAttr(trainable_statistics);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    kernel_context.EmplaceBackOutput(kernel_out_3);
    kernel_context.EmplaceBackOutput(kernel_out_4);
    kernel_context.EmplaceBackOutput(kernel_out_5);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (batch_norm_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor cast(const Tensor& x, DataType index_dtype, DataType value_dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "cast api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("cast", kernel_data_type);
    }
    VLOG(6) << "cast api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::sparse::CastInferMeta(MakeMetaTensor(*input_x), index_dtype, value_dtype, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(index_dtype);
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "cast api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("cast", kernel_data_type);
    }
    VLOG(6) << "cast api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::sparse::CastInferMeta(MakeMetaTensor(*input_x), index_dtype, value_dtype, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(index_dtype);
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (cast) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor conv3d(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, kernel);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(kernel.impl().get())) {

    VLOG(6) << "conv3d api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "conv3d_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("conv3d", kernel_data_type);
    }
    VLOG(6) << "conv3d api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_kernel = PrepareDataForDenseTensorInSparse(kernel);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::sparse::Conv3dInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_kernel), paddings, dilations, strides, groups, subm, key, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_kernel.get());
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackAttr(groups);
    kernel_context.EmplaceBackAttr(subm);
    kernel_context.EmplaceBackAttr(key);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "divide api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide", kernel_data_type);
    }
    VLOG(6) << "divide api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "divide api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide", kernel_data_type);
    }
    VLOG(6) << "divide api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor divide_scalar(const Tensor& x, float scalar) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "divide_scalar api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_scalar_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide_scalar", kernel_data_type);
    }
    VLOG(6) << "divide_scalar api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(scalar);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "divide_scalar api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_scalar_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("divide_scalar", kernel_data_type);
    }
    VLOG(6) << "divide_scalar api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(scalar);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide_scalar) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor expm1(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "expm1 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("expm1", kernel_data_type);
    }
    VLOG(6) << "expm1 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "expm1 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("expm1", kernel_data_type);
    }
    VLOG(6) << "expm1 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (expm1) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor isnan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "isnan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isnan_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("isnan", kernel_data_type);
    }
    VLOG(6) << "isnan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "isnan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isnan_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("isnan", kernel_data_type);
    }
    VLOG(6) << "isnan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (isnan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor leaky_relu(const Tensor& x, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "leaky_relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("leaky_relu", kernel_data_type);
    }
    VLOG(6) << "leaky_relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "leaky_relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("leaky_relu", kernel_data_type);
    }
    VLOG(6) << "leaky_relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (leaky_relu) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor log1p(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "log1p api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("log1p", kernel_data_type);
    }
    VLOG(6) << "log1p api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "log1p api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("log1p", kernel_data_type);
    }
    VLOG(6) << "log1p api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (log1p) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "multiply api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("multiply", kernel_data_type);
    }
    VLOG(6) << "multiply api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "multiply api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("multiply", kernel_data_type);
    }
    VLOG(6) << "multiply api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (multiply) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor pow(const Tensor& x, float factor) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "pow api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("pow", kernel_data_type);
    }
    VLOG(6) << "pow api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "pow api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("pow", kernel_data_type);
    }
    VLOG(6) << "pow api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (pow) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor relu(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu", kernel_data_type);
    }
    VLOG(6) << "relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu", kernel_data_type);
    }
    VLOG(6) << "relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor relu6(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "relu6 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu6", kernel_data_type);
    }
    VLOG(6) << "relu6 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "relu6 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("relu6", kernel_data_type);
    }
    VLOG(6) << "relu6 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu6) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "reshape api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "reshape_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape", kernel_data_type);
    }
    VLOG(6) << "reshape api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ReshapeInferMeta(MakeMetaTensor(*input_x), shape, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(shape));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "reshape api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "reshape_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape", kernel_data_type);
    }
    VLOG(6) << "reshape api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ReshapeInferMeta(MakeMetaTensor(*input_x), shape, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(shape));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (reshape) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor scale(const Tensor& x, float scale, float bias, bool bias_after_scale) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "scale api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("scale", kernel_data_type);
    }
    VLOG(6) << "scale api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(scale);
    kernel_context.EmplaceBackAttr(bias);
    kernel_context.EmplaceBackAttr(bias_after_scale);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "scale api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("scale", kernel_data_type);
    }
    VLOG(6) << "scale api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(scale);
    kernel_context.EmplaceBackAttr(bias);
    kernel_context.EmplaceBackAttr(bias_after_scale);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (scale) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sin(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "sin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sin", kernel_data_type);
    }
    VLOG(6) << "sin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "sin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sin", kernel_data_type);
    }
    VLOG(6) << "sin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sin) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sinh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "sinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sinh", kernel_data_type);
    }
    VLOG(6) << "sinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "sinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sinh", kernel_data_type);
    }
    VLOG(6) << "sinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sinh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor softmax(const Tensor& x, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "softmax api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("softmax", kernel_data_type);
    }
    VLOG(6) << "softmax api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "softmax api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("softmax", kernel_data_type);
    }
    VLOG(6) << "softmax api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (softmax) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sparse_coo_tensor(const Tensor& values, const Tensor& indices, const std::vector<int64_t>& shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(values);

  kernel_data_type = ParseDataType(values);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(values, indices);
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

  if (phi::DenseTensor::classof(values.impl().get()) && phi::DenseTensor::classof(indices.impl().get())) {

    VLOG(6) << "sparse_coo_tensor api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sparse_coo_tensor", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sparse_coo_tensor", kernel_data_type);
    }
    VLOG(6) << "sparse_coo_tensor api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_values = PrepareDataForDenseTensorInSparse(values);
    auto input_indices = PrepareDataForDenseTensorInSparse(indices);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::sparse::SparseCooTensorInferMeta(MakeMetaTensor(*input_values), MakeMetaTensor(*input_indices), shape, &meta_out);


    kernel_context.EmplaceBackInput(input_values.get());
    kernel_context.EmplaceBackInput(input_indices.get());
    kernel_context.EmplaceBackAttr(shape);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sparse_coo_tensor) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sqrt(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "sqrt api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sqrt", kernel_data_type);
    }
    VLOG(6) << "sqrt api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "sqrt api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sqrt", kernel_data_type);
    }
    VLOG(6) << "sqrt api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sqrt) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor square(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "square api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("square", kernel_data_type);
    }
    VLOG(6) << "square api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "square api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("square", kernel_data_type);
    }
    VLOG(6) << "square api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (square) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "subtract api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("subtract", kernel_data_type);
    }
    VLOG(6) << "subtract api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "subtract api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("subtract", kernel_data_type);
    }
    VLOG(6) << "subtract api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (subtract) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sum(const Tensor& x, const IntArray& axis, DataType dtype, bool keepdim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "sum api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sum_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sum", kernel_data_type);
    }
    VLOG(6) << "sum api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::SumInferMeta(MakeMetaTensor(*input_x), axis, dtype, keepdim, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axis));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackAttr(keepdim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "sum api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sum_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sum", kernel_data_type);
    }
    VLOG(6) << "sum api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::SumInferMeta(MakeMetaTensor(*input_x), axis, dtype, keepdim, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axis));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackAttr(keepdim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sum) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> sync_batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, mean, variance, scale, bias);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(mean.impl().get()) && phi::DenseTensor::classof(variance.impl().get()) && phi::DenseTensor::classof(scale.impl().get()) && phi::DenseTensor::classof(bias.impl().get())) {

    VLOG(6) << "sync_batch_norm_ api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sync_batch_norm_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("sync_batch_norm_", kernel_data_type);
    }
    VLOG(6) << "sync_batch_norm_ api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> api_output{Tensor(), mean, variance, Tensor(), Tensor(), Tensor()};
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_3 = SetSparseKernelOutput(&std::get<3>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_4 = SetSparseKernelOutput(&std::get<4>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_5 = SetSparseKernelOutput(&std::get<5>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_mean = PrepareDataForDenseTensorInSparse(mean);
    auto input_variance = PrepareDataForDenseTensorInSparse(variance);
    auto input_scale = PrepareDataForDenseTensorInSparse(scale);
    auto input_bias = PrepareDataForDenseTensorInSparse(bias);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_5(kernel_out_5, kernel_result.is_stride_kernel);

  phi::BatchNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_mean), MakeMetaTensor(*input_variance), MakeMetaTensor(*input_scale), MakeMetaTensor(*input_bias), is_test, momentum, epsilon, data_format, use_global_stats, trainable_statistics, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_mean.get());
    kernel_context.EmplaceBackInput(input_variance.get());
    kernel_context.EmplaceBackInput(input_scale.get());
    kernel_context.EmplaceBackInput(input_bias.get());
    kernel_context.EmplaceBackAttr(is_test);
    kernel_context.EmplaceBackAttr(momentum);
    kernel_context.EmplaceBackAttr(epsilon);
    kernel_context.EmplaceBackAttr(data_format);
    kernel_context.EmplaceBackAttr(use_global_stats);
    kernel_context.EmplaceBackAttr(trainable_statistics);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    kernel_context.EmplaceBackOutput(kernel_out_3);
    kernel_context.EmplaceBackOutput(kernel_out_4);
    kernel_context.EmplaceBackOutput(kernel_out_5);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sync_batch_norm_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor tan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "tan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tan", kernel_data_type);
    }
    VLOG(6) << "tan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "tan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tan", kernel_data_type);
    }
    VLOG(6) << "tan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor tanh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "tanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tanh", kernel_data_type);
    }
    VLOG(6) << "tanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "tanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("tanh", kernel_data_type);
    }
    VLOG(6) << "tanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tanh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor to_dense(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "to_dense api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_to_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_dense", kernel_data_type);
    }
    VLOG(6) << "to_dense api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "to_dense api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "csr_to_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_dense", kernel_data_type);
    }
    VLOG(6) << "to_dense api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (to_dense) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor to_sparse_coo(const Tensor& x, int64_t sparse_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (phi::DenseTensor::classof(x.impl().get())) {

    VLOG(6) << "to_sparse_coo api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "dense_to_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_sparse_coo", kernel_data_type);
    }
    VLOG(6) << "to_sparse_coo api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForDenseTensorInSparse(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(sparse_dim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "to_sparse_coo api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "csr_to_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_sparse_coo", kernel_data_type);
    }
    VLOG(6) << "to_sparse_coo api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(sparse_dim);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (to_sparse_coo) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor to_sparse_csr(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (phi::DenseTensor::classof(x.impl().get())) {

    VLOG(6) << "to_sparse_csr api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "dense_to_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_sparse_csr", kernel_data_type);
    }
    VLOG(6) << "to_sparse_csr api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForDenseTensorInSparse(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "to_sparse_csr api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_to_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("to_sparse_csr", kernel_data_type);
    }
    VLOG(6) << "to_sparse_csr api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (to_sparse_csr) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor transpose(const Tensor& x, const std::vector<int>& perm) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "transpose api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "transpose_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("transpose", kernel_data_type);
    }
    VLOG(6) << "transpose api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::TransposeInferMeta(MakeMetaTensor(*input_x), perm, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(perm);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "transpose api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "transpose_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("transpose", kernel_data_type);
    }
    VLOG(6) << "transpose api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::TransposeInferMeta(MakeMetaTensor(*input_x), perm, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(perm);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (transpose) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor values(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "values api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "values_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("values", kernel_data_type);
    }
    VLOG(6) << "values api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::sparse::ValuesInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "values api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "values_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("values", kernel_data_type);
    }
    VLOG(6) << "values api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::sparse::ValuesInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (values) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, x, y);
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

  if (phi::DenseTensor::classof(input.impl().get()) && x.is_sparse_csr_tensor() && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm", kernel_data_type);
    }
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_input = PrepareDataForDenseTensorInSparse(input);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_input), &meta_out);

    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (input.is_sparse_csr_tensor() && x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm", kernel_data_type);
    }
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_input = PrepareDataForSparseCsrTensor(input);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_input), &meta_out);

    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (phi::DenseTensor::classof(input.impl().get()) && x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm", kernel_data_type);
    }
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_input = PrepareDataForDenseTensorInSparse(input);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_input), &meta_out);

    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (input.is_sparse_coo_tensor() && x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("addmm", kernel_data_type);
    }
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_input = PrepareDataForSparseCooTensor(input);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_input), &meta_out);

    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

    kernel_context.EmplaceBackInput(input_input.get());
    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (addmm) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor coalesce(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "coalesce api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coalesce_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("coalesce", kernel_data_type);
    }
    VLOG(6) << "coalesce api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (coalesce) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  kernel_data_type = ParseDataType(dtype);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "full_like api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "full_like_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("full_like", kernel_data_type);
    }
    VLOG(6) << "full_like api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::Scalar(value));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "full_like api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "full_like_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("full_like", kernel_data_type);
    }
    VLOG(6) << "full_like api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::Scalar(value));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (full_like) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor fused_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(sparse_mask);

  kernel_data_type = ParseDataType(query);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, sparse_mask, key_padding_mask, attn_mask);
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

  if (phi::DenseTensor::classof(query.impl().get()) && phi::DenseTensor::classof(key.impl().get()) && phi::DenseTensor::classof(value.impl().get()) && sparse_mask.is_sparse_csr_tensor() && (!key_padding_mask || phi::DenseTensor::classof(key_padding_mask->impl().get())) && (!attn_mask || phi::DenseTensor::classof(attn_mask->impl().get()))) {

    VLOG(6) << "fused_attention api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "fused_attention_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_attention", kernel_data_type);
    }
    VLOG(6) << "fused_attention api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::SPARSE_CSR);
    auto input_query = PrepareDataForDenseTensorInSparse(query);
    auto input_key = PrepareDataForDenseTensorInSparse(key);
    auto input_value = PrepareDataForDenseTensorInSparse(value);
    auto input_sparse_mask = PrepareDataForSparseCsrTensor(sparse_mask);
    auto input_key_padding_mask = PrepareDataForDenseTensorInSparse(key_padding_mask);
    auto input_attn_mask = PrepareDataForDenseTensorInSparse(attn_mask);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::sparse::FusedAttentionInferMeta(MakeMetaTensor(*input_query), MakeMetaTensor(*input_key), MakeMetaTensor(*input_value), MakeMetaTensor(*input_sparse_mask), MakeMetaTensor(input_key_padding_mask), MakeMetaTensor(input_attn_mask), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    kernel_context.EmplaceBackInput(input_query.get());
    kernel_context.EmplaceBackInput(input_key.get());
    kernel_context.EmplaceBackInput(input_value.get());
    kernel_context.EmplaceBackInput(input_sparse_mask.get());
    kernel_context.EmplaceBackInput(key_padding_mask ? &(*input_key_padding_mask) : nullptr);
    kernel_context.EmplaceBackInput(attn_mask ? &(*input_attn_mask) : nullptr);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor masked_matmul(const Tensor& x, const Tensor& y, const Tensor& mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, mask);
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

  if (phi::DenseTensor::classof(x.impl().get()) && phi::DenseTensor::classof(y.impl().get()) && mask.is_sparse_csr_tensor()) {

    VLOG(6) << "masked_matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "masked_matmul_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("masked_matmul", kernel_data_type);
    }
    VLOG(6) << "masked_matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForDenseTensorInSparse(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), false, false, &meta_out);

    auto input_mask = PrepareDataForSparseCsrTensor(mask);

    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackInput(input_mask.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (masked_matmul) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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

  if (x.is_sparse_csr_tensor() && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul", kernel_data_type);
    }
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), false, false, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && y.is_sparse_csr_tensor()) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul", kernel_data_type);
    }
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_y = PrepareDataForSparseCsrTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), false, false, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul", kernel_data_type);
    }
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForDenseTensorInSparse(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), false, false, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_coo_tensor() && y.is_sparse_coo_tensor()) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("matmul", kernel_data_type);
    }
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_y = PrepareDataForSparseCooTensor(y);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), false, false, &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_y.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (matmul) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor maxpool(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "maxpool api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "maxpool_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("maxpool", kernel_data_type);
    }
    VLOG(6) << "maxpool api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::sparse::Pool3dInferMeta(MakeMetaTensor(*input_x), kernel_sizes, paddings, dilations, strides, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(kernel_sizes);
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, vec);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(vec.impl().get())) {

    VLOG(6) << "mv api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("mv", kernel_data_type);
    }
    VLOG(6) << "mv api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_vec = PrepareDataForDenseTensorInSparse(vec);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_vec), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_vec.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor() && phi::DenseTensor::classof(vec.impl().get())) {

    VLOG(6) << "mv api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("mv", kernel_data_type);
    }
    VLOG(6) << "mv api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCsrTensor(x);
    auto input_vec = PrepareDataForDenseTensorInSparse(vec);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::MvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_vec), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_vec.get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (mv) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor slice(const Tensor& x, const IntArray& axes, const IntArray& starts, const IntArray& ends) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "slice api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "slice_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("slice", kernel_data_type);
    }
    VLOG(6) << "slice api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axes));
    kernel_context.EmplaceBackAttr(phi::IntArray(starts));
    kernel_context.EmplaceBackAttr(phi::IntArray(ends));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.is_sparse_csr_tensor()) {

    VLOG(6) << "slice api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "slice_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("slice", kernel_data_type);
    }
    VLOG(6) << "slice api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
    auto input_x = PrepareDataForSparseCsrTensor(x);

  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(phi::IntArray(axes));
    kernel_context.EmplaceBackAttr(phi::IntArray(starts));
    kernel_context.EmplaceBackAttr(phi::IntArray(ends));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (slice) for input tensors is unimplemented, please check the type of input tensors."));
}


}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
