#include "paddle/phi/kernels/fused_act_dequant_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"


namespace phi {

template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

__global__ void FusedActDequant(
    const phi::float8_e4m3fn*__restrict__ Xin,
    const float *__restrict__ Xscale,
    phi::bfloat16*__restrict__ out,
    const int rows,
    const int cols
 ) {
  const int this_row_idx = blockIdx.x;
  if (this_row_idx >= rows) return;

  const int Xscale_stride = (cols + 127) / 128;  // 计算缩放因子的步长

  const int vector_size = 16;  // 向量的元素数量，处理16个元素

  // 每行的向量数量
  const int num_vectors = cols / vector_size;
  const int remaining_elements = cols % vector_size;

  const int tid = threadIdx.x;

  for (int vec_idx = tid; vec_idx < num_vectors; vec_idx += blockDim.x) {
    int x_offset = vec_idx * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;

    // 加载16个 __nv_fp8_e4m3 元素到向量中
    const VectorType<__nv_fp8_e4m3, vector_size>* X_vec_ptr =
      reinterpret_cast<const VectorType<__nv_fp8_e4m3, vector_size>*>(Xin + X_idx);
    VectorType<__nv_fp8_e4m3, vector_size> X_vec = X_vec_ptr[0];

    // 获取对应的缩放因子
    int64_t scale_idx = (int64_t)this_row_idx * (int64_t)Xscale_stride + (x_offset / 128);
    float this_scale = Xscale[scale_idx];

    // 初始化输出向量
    VectorType<__nv_bfloat16, vector_size> out_vec;

    // 逐元素处理向量中的数据
    #pragma unroll
    for (int i = 0; i < vector_size; ++i) {
      // 将fp8转换为float
      float X_value = static_cast<float>(X_vec.data[i]);
      // 乘以缩放因子
      X_value *= this_scale;
      // 转换为bfloat16并存储
      out_vec.data[i] = __float2bfloat16(X_value);
    }

    // 将输出向量存储到全局内存
    VectorType<__nv_bfloat16, vector_size>* out_vec_ptr =
        reinterpret_cast<VectorType<__nv_bfloat16, vector_size>*>(out + X_idx);
    out_vec_ptr[0] = out_vec;
  }

  // 处理剩余不能被向量化的元素
  if (remaining_elements > 0) {
    int x_offset = num_vectors * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;
    int64_t idx = X_idx + tid;
    if (tid < remaining_elements) {
      float X_value = static_cast<float>(Xin[idx]);
      X_value *= Xscale[(int64_t)this_row_idx * (int64_t)Xscale_stride + (x_offset / 128)];
      out[idx] = __float2bfloat16(X_value);
    }
  }
}

template <typename T, typename Context>
void FusedActDequantKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& x_scale,
                           DenseTensor* out) {
  
  // 获取维度
  auto x_dims = x.dims();
  int rows = x_dims[0];
  int cols = x_dims[1];
  
  // 分配输出内存
  out->Resize({rows, cols});
  dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
  
  auto out_ptr = reinterpret_cast<void*>(out->template data<phi::dtype::bfloat16>());
  cudaMemsetAsync(out_ptr,
                  0,
                  sizeof(phi::dtype::bfloat16) * rows * cols,
                  dev_ctx.stream());
  
  // 直接调用CUDA kernel
  dim3 grid(rows);
  dim3 block(256);
  
  FusedActDequant<<<grid, block, 0, dev_ctx.stream()>>>(
      x.data<phi::dtype::float8_e4m3fn>(),
      x_scale.data<float>(),
      out->data<phi::dtype::bfloat16>(),
      rows,
      cols);
  
  #ifdef PADDLE_WITH_CUDA_CHECK
  auto cuda_error = cudaGetLastError();
  PADDLE_ENFORCE_GPU_SUCCESS(cuda_error);
  #endif
}

}

PD_REGISTER_KERNEL(fused_act_dequant,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedActDequantKernel,
                   phi::dtype::float8_e4m3fn,
                   float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BFLOAT16);
}