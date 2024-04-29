
#include "paddle/phi/api/include/strings_api.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/strings/nullary.h"
#include "paddle/phi/infermeta/strings/unary.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle {
namespace experimental {
namespace strings {


PADDLE_API Tensor empty(const IntArray& shape, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::PSTRING_UNION;
  DataType kernel_data_type = DataType::PSTRING;

  kernel_backend = ParseBackend(place);


  // 1. Get kernel signature and kernel
  VLOG(6) << "empty api strings kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_empty", {kernel_backend, kernel_layout, kernel_data_type});
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("empty", kernel_data_type);
  }
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "empty api strings kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
  

  //  3. Set output
  
  Tensor api_output;
  phi::StringTensor* kernel_out = dynamic_cast<phi::StringTensor*>(SetStringsKernelOutput(&api_output, TensorType::STRING_TENSOR));
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::strings::CreateInferMeta(shape, &meta_out);


  // 4. run kernel

  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::IntArray&, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, phi::IntArray(shape), kernel_out);

  return api_output;
}

PADDLE_API Tensor empty_like(const Tensor& x, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::PSTRING_UNION;
  DataType kernel_data_type = DataType::PSTRING;

  kernel_backend = ParseBackendWithInputOrder(place, x);

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  kernel_backend = kernel_key.backend();

  // 1. Get kernel signature and kernel
  VLOG(6) << "empty_like api strings kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_empty_like", {kernel_backend, kernel_layout, kernel_data_type});
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("empty_like", kernel_data_type);
  }
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "empty_like api strings kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
  
  auto input_x = TensorToStringTensor(x);

  //  3. Set output
  
  Tensor api_output;
  phi::StringTensor* kernel_out = dynamic_cast<phi::StringTensor*>(SetStringsKernelOutput(&api_output, TensorType::STRING_TENSOR));
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::strings::CreateLikeInferMeta(MakeMetaTensor(*input_x), &meta_out);


  // 4. run kernel

  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::StringTensor&, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, kernel_out);

  return api_output;
}

PADDLE_API Tensor lower(const Tensor& x, bool use_utf8_encoding) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::PSTRING_UNION;
  DataType kernel_data_type = DataType::PSTRING;

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  kernel_backend = kernel_key.backend();

  // 1. Get kernel signature and kernel
  VLOG(6) << "lower api strings kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_lower", {kernel_backend, kernel_layout, kernel_data_type});
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("lower", kernel_data_type);
  }
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lower api strings kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
  
  auto input_x = TensorToStringTensor(x);

  //  3. Set output
  
  Tensor api_output;
  phi::StringTensor* kernel_out = dynamic_cast<phi::StringTensor*>(SetStringsKernelOutput(&api_output, TensorType::STRING_TENSOR));
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::strings::CreateLikeInferMeta(MakeMetaTensor(*input_x), &meta_out);


  // 4. run kernel

  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::StringTensor&, bool, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, use_utf8_encoding, kernel_out);

  return api_output;
}

PADDLE_API Tensor upper(const Tensor& x, bool use_utf8_encoding) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::PSTRING_UNION;
  DataType kernel_data_type = DataType::PSTRING;

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  kernel_backend = kernel_key.backend();

  // 1. Get kernel signature and kernel
  VLOG(6) << "upper api strings kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_upper", {kernel_backend, kernel_layout, kernel_data_type});
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("upper", kernel_data_type);
  }
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "upper api strings kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
  
  auto input_x = TensorToStringTensor(x);

  //  3. Set output
  
  Tensor api_output;
  phi::StringTensor* kernel_out = dynamic_cast<phi::StringTensor*>(SetStringsKernelOutput(&api_output, TensorType::STRING_TENSOR));
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::strings::CreateLikeInferMeta(MakeMetaTensor(*input_x), &meta_out);


  // 4. run kernel

  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::StringTensor&, bool, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, use_utf8_encoding, kernel_out);

  return api_output;
}


}  // namespace strings
}  // namespace experimental
}  // namespace paddle
