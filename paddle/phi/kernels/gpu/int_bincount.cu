#include "paddle/extension.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/common/flags.h"
#include <vector>
#include <cstdint>
// #include "cub/device/device_histogram.cuh"

namespace phi {

COMMON_DECLARE_bool(enable_pir_api);


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

// template <typename T, typename BinsT>
// void IntBincountImpl(const T *x, int64_t n, T min_v, T max_v, BinsT *bins, cudaStream_t stream, phi::Place place) {
//   paddle::Tensor workspace;
//   void *workspace_ptr = nullptr;
//   size_t workspace_size = 0;
// #pragma unroll
//   for (int i = 0; i < 2; ++i) {
//     if (workspace_size > 0) {
//       workspace = paddle::empty({static_cast<int64_t>(workspace_size)}, paddle::DataType::UINT8, place);
//       workspace_ptr = workspace.data();
//     }
//     auto err = cub::DeviceHistogram::HistogramEven(
//       workspace_ptr, workspace_size, x, bins, max_v - min_v + 1, min_v, max_v, n, stream);
//     PD_CHECK(err == cudaSuccess, "HistogramEven error: %s", cudaGetErrorString(err));
//   }
// }

template <typename T, typename BinsT>
__global__ void histogram_smem_kernel(const T* x,
                                      int64_t n,
                                      T min_v,
                                      T max_v,
                                      BinsT* partial_hist) {
  extern __shared__ unsigned int smem[];                  // 每个 block 的局部直方图
  int bin_count = static_cast<int>(max_v - min_v + 1);
  int tid_in_block = threadIdx.x;
  int block_threads = blockDim.x;
  int block_id = blockIdx.x;
  int global_tid = block_id * blockDim.x + tid_in_block;
  int total_threads = blockDim.x * gridDim.x;

  for (int i = tid_in_block; i < bin_count; i += block_threads) {
    smem[i] = 0u;
  }
  __syncthreads();

  for (int idx = global_tid; idx < n; idx += total_threads) {
    T v = x[idx];
    if (v >= min_v && v <= max_v) {
      int b = static_cast<int>(v - min_v);
      atomicAdd(&smem[b], 1u);
    }
  }
  __syncthreads();

  BinsT* out = partial_hist + block_id * bin_count;
  for (int i = tid_in_block; i < bin_count; i += block_threads) {
    out[i] = static_cast<BinsT>(smem[i]);
  }
}

template <typename BinsT>
__global__ void histogram_accum_kernel(const BinsT* partial_hist,
                                       int num_blocks,
                                       int bin_count,
                                       BinsT* hist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bin_count) return;
  BinsT sum = 0;
  for (int b = 0; b < num_blocks; ++b) {
    sum += partial_hist[b * bin_count + i];
  }
  hist[i] = sum;
}

void IntBincountImpl(const T *x, int64_t n,
                     T min_v, T max_v,
                     BinsT *bins,
                     cudaStream_t stream,
                     phi::Place place) {
  int bin_count   = static_cast<int>(max_v - min_v + 1);
  const int threads = 256;
  int num_blocks = (n + threads - 1) / threads;
  // 1) 分配 partial_hist：shape [num_blocks, bin_count]
  auto partial = paddle::empty({num_blocks, bin_count},
                               TransToDataType(sizeof(BinsT)==4 ? phi::DataType::UINT32
                                                                 : phi::DataType::UINT64),
                               place);
  BinsT* partial_ptr = partial.data<BinsT>();

  // 2) Phase 1：Block 局部直方图
  size_t smem_size = bin_count * sizeof(unsigned int);
  histogram_smem_kernel<T,BinsT>
    <<<num_blocks, threads, smem_size, stream>>>(x, n, min_v, max_v, partial_ptr);

  // 3) Phase 2：合并所有局部直方图
  int out_threads = 256;
  int out_blocks  = (bin_count + out_threads - 1) / out_threads;
  histogram_accum_kernel<BinsT>
    <<<out_blocks, out_threads, 0, stream>>>(partial_ptr, num_blocks, bin_count, bins);
}

std::vector<paddle::Tensor> IntBincount(const paddle::Tensor &x, int64_t low, int64_t high, int64_t out_dtype) {
  PD_CHECK(low < high);
  auto bins_width = high - low;
  PD_CHECK(bins_width + 1 < std::numeric_limits<int>::max());

  auto bins_dtype = TransToDataType(out_dtype);
  auto place = x.place();
  auto bins = paddle::empty({bins_width}, bins_dtype, place);
  auto stream = x.stream();

  PD_DISPATCH_INTEGRAL_TYPES(x.dtype(), "int_bin_count_dispatch", ([&] {
    auto low_v = static_cast<data_t>(low);
    auto high_v = static_cast<data_t>(high);
    PD_CHECK(static_cast<int64_t>(low_v) == low);
    PD_CHECK(static_cast<int64_t>(high_v) == high);
    const auto *x_data = x.data<data_t>();
    void *bins_data = bins.data();
    int64_t n = x.numel();
    if (bins_dtype == paddle::DataType::INT32) {
      IntBincountImpl<data_t, uint32_t>(x_data, n, low_v, high_v, static_cast<uint32_t *>(bins_data), stream, place); 
    } else if (bins_dtype == paddle::DataType::INT64) {
      using ULLI = unsigned long long int;
      static_assert(sizeof(int64_t) == sizeof(ULLI)); 
      IntBincountImpl<data_t, ULLI>(x_data, n, low_v, high_v, static_cast<ULLI *>(bins_data), stream, place);
    } else {
      PD_THROW("Only support INT32 and INT64, but got %s", bins_dtype);
    }
  }));

  return {bins};
}
}

PD_BUILD_OP(int_bincount)
    .Inputs({"x"})
    .Outputs({"y"})
    .Attrs({"low: int64_t", "high: int64_t", "dtype: int64_t"})
    .SetKernelFn(PD_KERNEL(phi::IntBincount))
    .SetInferShapeFn(PD_INFER_SHAPE(phi::IntBincountInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(phi::IntBincountInferDType));