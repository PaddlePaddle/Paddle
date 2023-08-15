#include "paddle/phi/kernels/gpudnn/conv_mudnn_helper.h"

#include "glog/logging.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ConvCudnnKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings_t,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations_t,
                     int groups,
                     const std::string& data_format,
                     DenseTensor* output) {
  std::cout << "Conv start " << __FILE__ << " :" << __LINE__ << std::endl;
  ctx.template Alloc<T>(output);
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  auto compute_format = phi::backends::gpu::DataLayout::kNCHW;

  auto input_shape = input.dims();
  auto weight_shape = filter.dims();

  auto in = CreateMUTensor(input);
  auto out = CreateMUTensor(*output);
  auto ke = CreateMUTensor(filter);

  ConfigFormat(in, compute_format);
  ConfigFormat(out, compute_format);
  ConfigFormat(ke, compute_format);

  auto handle = ctx.cudnn_handle();
  std::cout << handle->GetDeviceId() << " " << __FILE__ << std::endl;
  ::musa::dnn::Convolution c;
  ConfigConv(c, strides, paddings, dilations, groups);
  ::musa::dnn::Convolution::Algorithm algo;
  c.GetRecommendForwardAlgorithm(*handle, algo, out, in, ke);
  c.Run(*handle, out, in, ke, algo, InternalMemAlloc);
  std::cout << "Conv end " << __FILE__ << " :" << __LINE__ << std::endl;
}

} // namespace phi

PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   phi::dtype::float16) {}
