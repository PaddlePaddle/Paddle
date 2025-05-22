// #include "paddle/extension.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/common/flags.h"
#include <vector>
#include <cstdint>
#include "cub/device/device_histogram.cuh"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(enable_pir_api);

namespace phi{
static paddle::DataType TransToDataType(int64_t dtype) {
  if (FLAGS_enable_pir_api) {
    return static_cast<paddle::DataType>(dtype);
  } else {
    return phi::TransToPhiDataType(dtype);
  }
}

std::vector<std::vector<int64_t>> IntBincountInferShape(
    std::vector<int64_t> x_shape,
    int64_t min_value,
    int64_t max_value,
    int64_t out_dtype) {
  return {{max_value - min_value}};
}

std::vector<paddle::DataType> IntBincountInferDType(
    paddle::DataType x_dtype,
    int64_t min_value,
    int64_t max_value,
    int64_t out_dtype) {
  return {TransToDataType(out_dtype)};
}

template <typename T, typename BinsT, typename Context>
void IntBincountImpl(const Context& ctx, const T *x, int64_t n, T min_v, T max_v, BinsT *bins, phi::Place place) {
  DenseTensor workspace;
  void *workspace_ptr = nullptr;
  size_t workspace_size = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (workspace_size > 0) {
      workspace = phi::EmptyLike<paddle::DataType::UINT8>(ctx, {static_cast<int64_t>(workspace_size)});
      workspace_ptr = workspace.data();
    }
    auto err = cub::DeviceHistogram::HistogramEven(
      workspace_ptr, workspace_size, x, bins, max_v - min_v + 1, min_v, max_v, n, ctx.stream());
    PD_CHECK(err == cudaSuccess, "HistogramEven error: %s", cudaGetErrorString(err));
  }
}

template<typename Context>
std::vector<DenseTensor> IntBincount(const Context& ctx, const DenseTensor &x, int64_t low, int64_t high, int64_t out_dtype) {
  PD_CHECK(low < high);
  auto bins_width = high - low;
  PD_CHECK(bins_width + 1 < std::numeric_limits<int>::max());

  auto bins_dtype = TransToDataType(out_dtype);
  auto place = x.place();
//   auto bins = phi::Empty({bins_width}, bins_dtype, place);
  DenseTensor bins = phi::EmptyLike<bins_dtype>(ctx, {bins_width});

  PD_DISPATCH_INTEGRAL_TYPES(x.dtype(), "int_bin_count_dispatch", ([&] {
    auto low_v = static_cast<data_t>(low);
    auto high_v = static_cast<data_t>(high);
    PD_CHECK(static_cast<int64_t>(low_v) == low);
    PD_CHECK(static_cast<int64_t>(high_v) == high);
    const auto *x_data = x.data<data_t>();
    void *bins_data = bins.data();
    int64_t n = x.numel();
    if (bins_dtype == paddle::DataType::INT32) {
      IntBincountImpl<data_t, uint32_t>(x_data, n, low_v, high_v, static_cast<uint32_t *>(bins_data), ctx.stream(), place); 
    } else if (bins_dtype == paddle::DataType::INT64) {
      using ULLI = unsigned long long int;
      static_assert(sizeof(int64_t) == sizeof(ULLI)); 
      IntBincountImpl<data_t, ULLI>(x_data, n, low_v, high_v, static_cast<ULLI *>(bins_data), ctx.stream(), place);
    } else {
      PD_THROW("Only support INT32 and INT64, but got %s", bins_dtype);
    }
  }));

  return {bins};
}
} // namespace phi

// PD_BUILD_OP(int_bincount)
//     .Inputs({"x"})
//     .Outputs({"y"})
//     .Attrs({"low: int64_t", "high: int64_t", "dtype: int64_t"})
//     .SetKernelFn(PD_KERNEL(phi::IntBincount))
//     .SetInferShapeFn(PD_INFER_SHAPE(phi::IntBincountInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(phi::IntBincountInferDType));

PD_REGISTER_KERNEL(int_bincount,
                   GPU,
                   ALL_LAYOUT,
                   phi::IntBincount,
                   int64_t,
                   uint32_t) {}
