#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cooperative_groups.h>
using namespace std;

using float16 = half;
using bfloat16 = nv_bfloat16;
using float8e4m3 = __nv_fp8_e4m3;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define DIVISIBLE(a, b) ((a) % (b) == 0)

__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                \
  __device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
    DTYPE tmp_val = value;                                                                \
    unsigned int mask = __activemask();                                                   \
    unsigned int lane = __popc(mask);                                                     \
    if (lane < 32) {                                                                      \
      for (int offset = 16; offset > 0; offset >>= 1) {                                   \
        DTYPE shfl_res = __shfl_down_sync(mask, tmp_val, offset);                         \
        if ((threadIdx.x & 0x1f) + offset >= lane) {                                      \
          shfl_res = (DTYPE)(INITIAL_VALUE);                                              \
        }                                                                                 \
        tmp_val = cinn_##REDUCE_TYPE(tmp_val, shfl_res);                                  \
      }                                                                                   \
    } else {                                                                              \
      for (int offset = 16; offset > 0; offset >>= 1) {                                   \
        tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_xor_sync(mask, tmp_val, offset));    \
      }                                                                                   \
    }                                                                                     \
    return tmp_val;                                                                       \
  }

#define CINN_BLOCK_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_internal)  \
  DTYPE tmp_val = cinn_warp_shuffle_internal(value);               \
  if (return_warp || blockDim.x <= 32) {                           \
    return tmp_val;                                                \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x % 32 == 0) {                                     \
    shm[threadIdx.x / 32] = tmp_val;                               \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x < (blockDim.x + 31) / 32) {                      \
    tmp_val = cinn_warp_shuffle_internal(shm[threadIdx.x]);        \
    if (threadIdx.x == 0) {                                        \
      shm[0] = tmp_val;                                            \
    }                                                              \
  }                                                                \
  __syncthreads();                                                 \
  return shm[0];

#define CINN_BLOCK_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE value, DTYPE* shm, bool return_warp = false) { \
    CINN_BLOCK_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }

#define CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value)                          \
  int tid = threadIdx.y * blockDim.x + threadIdx.x;                            \
  __syncthreads();                                                             \
  shm[tid] = value;                                                            \
  __syncthreads();                                                             \
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {                \
    if (threadIdx.y < offset) {                                                \
      shm[tid] = cinn_##REDUCE_TYPE(shm[tid], shm[tid + offset * blockDim.x]); \
    }                                                                          \
    __syncthreads();                                                           \
  }                                                                            \
  return shm[threadIdx.x];

#define CINN_DISCRETE_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)          \
  __device__ inline DTYPE cinn_discrete_reduce_##REDUCE_TYPE(const DTYPE value, DTYPE* shm) { \
    CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value);                             \
  }

#define CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, init_value, DTYPE)                       \
  cooperative_groups::this_grid().sync();                                           \
  DTYPE tmp_val = init_value;                                                       \
  for (int y = 0; y < gridDim.y; y++) {                                             \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, mem[y * spatial_size + spatial_index]); \
  }                                                                                 \
  return tmp_val;

#define CINN_GRID_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                   \
  __device__ inline DTYPE cinn_grid_reduce_##REDUCE_TYPE(const DTYPE* mem, int spatial_size, int spatial_index) { \
    CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, (DTYPE)(INITIAL_VALUE), DTYPE);              \
  }

#define cinn_sum_fp32(a, b) ((a) + (b))
CINN_WARP_SHUFFLE_INTERNAL_IMPL(sum_fp32, 0.0f, float)
CINN_BLOCK_REDUCE_MACRO(sum_fp32, 0.0f, float)
CINN_DISCRETE_REDUCE_MACRO(sum_fp32, 0.0f, float)
CINN_GRID_REDUCE_MACRO(sum_fp32, 0.0f, float)

#define cinn_sum_fp16(a, b) ((a) + (b))
CINN_WARP_SHUFFLE_INTERNAL_IMPL(sum_fp16, (float16)0.0, float16)
CINN_BLOCK_REDUCE_MACRO(sum_fp16, (float16)0.0, float16)
CINN_DISCRETE_REDUCE_MACRO(sum_fp16, (float16)0.0, float16)
CINN_GRID_REDUCE_MACRO(sum_fp16, (float16)0.0, float16)

#define cinn_max_fp32(a, b) max((a), (b))
#define FLT_MAX	3.40282347e+38f
CINN_WARP_SHUFFLE_INTERNAL_IMPL(max_fp32, -FLT_MAX, float)
CINN_BLOCK_REDUCE_MACRO(max_fp32, -FLT_MAX, float)
CINN_DISCRETE_REDUCE_MACRO(max_fp32, -FLT_MAX, float)
CINN_GRID_REDUCE_MACRO(max_fp32, -FLT_MAX, float)

__device__ inline float cinn_nvgpu_log_fp32(float x) { return logf(x); }      // log(x)
__device__ inline float cinn_nvgpu_sqrt_fp32(float x) { return sqrtf(x); }     // sqrt(x)
__device__ inline float cinn_nvgpu_rsqrt_fp32(float x) { return rsqrtf(x); }  // 1/sqrt(x)
__device__ inline float cinn_nvgpu_exp_fp32(float x) { return expf(x); }       // exp(x)
__device__ inline float cinn_nvgpu_pow_fp32(float x, float y) { return powf(x, y); }  // x^y
__device__ inline float cinn_nvgpu_ceil_fp32(float x) { return ceilf(x); }     // ceil(x)
__device__ inline float cinn_nvgpu_log2_fp32(float x) { return log2f(x); }     // log2(x)
__device__ inline int cinn_nvgpu_bitwise_xor_int32(int a, int b) { return a ^ b; }
// __device__ inline float cinn_nvgpu_rcp_fp32(float x) { return __frcp_rn(x); }  // 1/x
__device__ inline float cinn_nvgpu_abs_fp32(float x) { return fabsf(x); }  // abs(x)
__device__ inline float16 cinn_nvgpu_abs_fp16(float16 x) {
    return __habs(x);  // abs(x)
}
__device__ inline int64_t cinn_nvgpu_bitwise_xor_int64(int64_t a, int64_t b) { return a ^ b; }

__device__ inline bool cinn_grid_reduce_update_semaphore(int *semaphores) {
  __shared__ bool done;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    int old = atomicAdd(&semaphores[blockIdx.x], 1);
    done = (old == (gridDim.y - 1));
  }
  __syncthreads();
  return done;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct GpuTimer {
  cudaEvent_t begin, end;
  GpuTimer() {
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
  }
  ~GpuTimer() {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(begin); }
  void stop() { cudaEventRecord(end); }
  float elapsed() {
    float ms;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, begin, end);
    return ms;
  }
  template <typename F>
  float trial(int n, F&& f) {
    start();
    for (int i = 0; i < n; i++) f();
    stop();
    return elapsed() / n;
  }
};

__device__ inline float16 max(float16 a, float16 b) { return __hmax(a, b); }
__device__ inline float16 min(float16 a, float16 b) { return __hmin(a, b); }

template <typename T>
struct Two { __device__ T operator()(int) { return T(2); } };
template <typename T>
struct Zero { __device__ T operator()(int) { return T(0); } };
struct Arange { __device__ int operator()(int i) { return i; } };
struct RandInit {
  unsigned long long seed;
  __device__ curandState operator()(int i) {
    curandState state;
    curand_init(seed, i, 0, &state);
    return state;
  }
};
struct Randn { curandState* state; };
struct Randint {
  curandState* state;
  int min, max;
};

template <typename T, typename F>
__global__ void vector_map(T* p, int n, F f) {
  for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
    int i = k + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      p[i] = f(i);
    }
  }
}
template <typename T>
__global__ void vector_map(T* p, int n, Randn f) {
  int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state = f.state[seq_id];
  for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
    int i = k + seq_id;
    if (i < n) {
      p[i] = curand_normal(&state);
    }
  }
  f.state[seq_id] = state;
}
template <typename T>
__global__ void vector_map(T* p, int n, Randint f) {
  int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state = f.state[seq_id];
  for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
    int i = k + seq_id;
    if (i < n) {
      float r = curand_uniform(&state);
      p[i] = f.min + (T)((f.max - f.min) * r);
      p[i] = 4 - i;
    }
  }
  f.state[seq_id] = state;
}

template <typename T>
struct Tensor {
  int numel;
  T *ptr;

  Tensor(int numel) : numel(numel) { init_storage(); }

  template <typename Init>
  Tensor(int numel, Init init) : Tensor(numel) { apply(init); }

  Tensor(const string& path) {
    ifstream fin(path, ios::binary | ios::ate);
    if (!fin.is_open()) {
      cerr << "Cannot open file: " << path << endl;
      exit(1);
    }

    streampos fsize = fin.tellg();
    if (fsize % sizeof(T)) {
      cerr << "File size is not a multiple of type size: " << path << endl;
      exit(1);
    }
    numel = fsize / sizeof(T);

    fin.seekg(0, ios::beg);
    char* hptr = new char[fsize];
    if (!fin.read(hptr, fsize)) {
        cerr << "Failed to read file: " << path << endl;
        delete[] hptr;
        exit(1);
    }
    fin.close();

    init_storage();

    cudaMemcpy(ptr, hptr, size(), cudaMemcpyHostToDevice);
    delete[] hptr;
  }

  void init_storage() {
    cudaError_t err = cudaMalloc((void **)&ptr, size());
    if (err != cudaSuccess) {  
      cerr << "error in allocating " << size() << " bytes: "
           << cudaGetErrorString(err) << endl;
      exit(1);
    }
  }

  size_t size() const { return sizeof(T) * numel; }

  void to_host() {
    T* host_ptr = new T[numel];
    cudaMemcpy(host_ptr, ptr, size(), cudaMemcpyDeviceToHost);
    cudaFree(ptr);
    ptr = host_ptr;
  }

  template <typename F>
  void apply(F f) { vector_map<<<64, 256>>>(ptr, numel, f); }

  void clear() { apply(Zero<T>()); }

  operator T*() { return ptr; }
  T** operator&() { return &ptr; }
  T operator[](int i) { return ptr[i]; }

  void compare(const Tensor<T>& o) {
    if (numel != o.numel) {
      cerr << "cannot compare tensors with different numels" << endl;
      exit(1);
    }
    int errcnt = 0;
    for (int i = 0; i < numel; i++) {
      float desired = ptr[i];
      float actual = o.ptr[i];
      if (abs(actual - desired) > 1e-6 + 1e-4 * abs(desired)) {
        if (++errcnt < 8) {
          cerr << i << " = " << desired << " " << actual << endl;
        }
      }
    }
    if (errcnt == 0) {
      cout << "ok" << endl;
    } else {
      float rate = ((float)errcnt / max(numel, 1));
      cerr << "num diff: " << errcnt << " / " << numel << " ("
           << rate * 100 << "%)" << endl;
    }
  }
};

#define PRINT_MAX_BLOCK_PER_SM(func, numThreads)     \
  do {                                               \
    int numBlocksPerSm = 0;                          \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(   \
        &numBlocksPerSm, func, numThreads, 0);       \
    cerr << #func << ": " << numBlocksPerSm << endl; \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ inline float cinn_nvgpu_rcp_fp32(float x) {
  float res;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(x));
  return res;
}

__device__ inline double cinn_nvgpu_rcp_fp64(double x) {
  double res;
  asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(res) : "d"(x));
  return res;
}

#define WELFORD_STRUCT_MACRO(TYPENAME, DTYPE) \
  struct TYPENAME {                           \
    DTYPE mean;                               \
    DTYPE m2;                                 \
    DTYPE weight;                             \
    __device__ TYPENAME() {};                 \
    __device__ explicit TYPENAME(DTYPE value) : mean(value), m2(0), weight(1) {} \
    __device__ TYPENAME(DTYPE mean, DTYPE m2, DTYPE weight) : mean(mean), m2(m2), weight(weight) {} \
    __device__ explicit operator DTYPE() const { return m2 / weight; } \
  };

#define WELFORD_REDUCE_MACRO(TYPENAME, DTYPE, RCP_FUNC) \
  __device__ inline TYPENAME operator+(const TYPENAME& a, const TYPENAME& b) { \
    DTYPE delta = b.mean - a.mean;                      \
    DTYPE weight = a.weight + (DTYPE)1;                 \
    DTYPE mean = a.mean + delta * RCP_FUNC(weight);     \
    DTYPE m2 = a.m2 + delta * (b.mean - mean);          \
    return {mean, m2, weight};                          \
  }

#define WELFORD_COMBINE_MACRO(TYPENAME, DTYPE, RCP_FUNC)           \
  __device__ inline TYPENAME cinn_sum_##TYPENAME(const TYPENAME& a, const TYPENAME& b) { \
    DTYPE delta = b.mean - a.mean;                                 \
    DTYPE weight = a.weight + b.weight;                            \
    DTYPE w2_over_w = b.weight * RCP_FUNC(max(weight, (DTYPE)1));  \
    DTYPE mean = a.mean + delta * w2_over_w;                       \
    DTYPE m2 = a.m2 + b.m2 + delta * delta * a.weight * w2_over_w; \
    return {mean, m2, weight};                                     \
  }

#define WELFORD_SHFL_SYNC_MACRO(TYPENAME, DTYPE, SHFL_FUNC, ARG2_TYPE, ARG2) \
  __device__ inline TYPENAME SHFL_FUNC(unsigned mask, const TYPENAME& var, ARG2_TYPE ARG2, int width = 32) { \
    DTYPE mean = SHFL_FUNC(mask, var.mean, ARG2, width);                     \
    DTYPE m2 = SHFL_FUNC(mask, var.m2, ARG2, width);                         \
    DTYPE weight = SHFL_FUNC(mask, var.weight, ARG2, width);                 \
    return {mean, m2, weight};                                               \
  }

#define EXPAND_WELFORD_MACRO(TYPE_SUFFIX, DTYPE)                                           \
  WELFORD_STRUCT_MACRO(welford_##TYPE_SUFFIX, DTYPE)                                       \
  WELFORD_REDUCE_MACRO(welford_##TYPE_SUFFIX, DTYPE, cinn_nvgpu_rcp_##TYPE_SUFFIX)         \
  WELFORD_COMBINE_MACRO(welford_##TYPE_SUFFIX, DTYPE, cinn_nvgpu_rcp_##TYPE_SUFFIX)        \
  WELFORD_SHFL_SYNC_MACRO(welford_##TYPE_SUFFIX, DTYPE, __shfl_down_sync, unsigned, delta) \
  WELFORD_SHFL_SYNC_MACRO(welford_##TYPE_SUFFIX, DTYPE, __shfl_xor_sync, int, laneMask)

EXPAND_WELFORD_MACRO(fp32, float)
EXPAND_WELFORD_MACRO(fp64, double)

CINN_WARP_SHUFFLE_INTERNAL_IMPL(sum_welford_fp32, welford_fp32(0.0f, 0.0f, 0), welford_fp32)
CINN_BLOCK_REDUCE_MACRO(sum_welford_fp32, welford_fp32(0.0f, 0.0f, 0), welford_fp32)
CINN_DISCRETE_REDUCE_MACRO(sum_welford_fp32, welford_fp32(0.0f, 0.0f, 0), welford_fp32)
CINN_GRID_REDUCE_MACRO(sum_welford_fp32, welford_fp32(0.0f, 0.0f, 0), welford_fp32)

CINN_WARP_SHUFFLE_INTERNAL_IMPL(sum_welford_fp64, welford_fp64(0.0f, 0.0f, 0), welford_fp64)
CINN_BLOCK_REDUCE_MACRO(sum_welford_fp64, welford_fp64(0.0f, 0.0f, 0), welford_fp64)
CINN_DISCRETE_REDUCE_MACRO(sum_welford_fp64, welford_fp64(0.0f, 0.0f, 0), welford_fp64)
CINN_GRID_REDUCE_MACRO(sum_welford_fp64, welford_fp64(0.0f, 0.0f, 0), welford_fp64)

#define CINN_ENTAIL_LOOP_CONDITION(__loop_var, __cond, __stride) \
  }                                                              \
  for (decltype(__stride) __loop_var = 0; __cond; __loop_var += __stride) {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define ARGIDX_STRUCT_MACRO(TYPENAME, DTYPE, ITYPE) \
  struct TYPENAME { \
    DTYPE value; \
    ITYPE index; \
    __device__ TYPENAME() {} \
    __device__ TYPENAME(DTYPE value, ITYPE index) : value(value), index(index) {} \
    __device__ explicit operator ITYPE() { return index; } \
  };

#define ARGIDX_SHFL_SYNC_MACRO(TYPENAME, DTYPE, ITYPE, SHFL_FUNC, ARG2_TYPE, ARG2) \
  __device__ inline TYPENAME SHFL_FUNC(unsigned mask, const TYPENAME& var, ARG2_TYPE ARG2, int width = 32) { \
    DTYPE value = SHFL_FUNC(mask, var.value, ARG2, width);                     \
    ITYPE index = SHFL_FUNC(mask, var.index, ARG2, width);                     \
    return {value, index};                                                     \
  }

#define ARGIDX_COMBINE_MACRO(TYPENAME) \
  __device__ TYPENAME cinn_min_##TYPENAME(TYPENAME a, TYPENAME b) { \
    return a.value < b.value ? a : b; \
  } \
  __device__ TYPENAME cinn_max_##TYPENAME(TYPENAME a, TYPENAME b) { \
    return a.value > b.value ? a : b; \
  } \
  __device__ TYPENAME min(TYPENAME a, TYPENAME b) { return cinn_min_##TYPENAME(a, b); } \
  __device__ TYPENAME max(TYPENAME a, TYPENAME b) { return cinn_max_##TYPENAME(a, b); }

#define ARGIDX_REDUCE_MACRO(TYPENAME, METHOD, DINIT) \
  CINN_WARP_SHUFFLE_INTERNAL_IMPL(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_BLOCK_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_DISCRETE_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_GRID_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME)

#define EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, ITYPE, INAME) \
  ARGIDX_STRUCT_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE) \
  ARGIDX_COMBINE_MACRO(argidx_##DNAME##_##INAME) \
  ARGIDX_SHFL_SYNC_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE, __shfl_down_sync, unsigned, delta) \
  ARGIDX_SHFL_SYNC_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE, __shfl_xor_sync, int, laneMask) \
  ARGIDX_REDUCE_MACRO(argidx_##DNAME##_##INAME, min, DMAX) \
  ARGIDX_REDUCE_MACRO(argidx_##DNAME##_##INAME, max, DMIN)

#define EXPAND_ARGIDX_DTYPE_MACRO(DTYPE, DNAME, DMIN, DMAX) \
  EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, int, i32) \
  EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, int64_t, i64)

EXPAND_ARGIDX_DTYPE_MACRO(float16, fp16, 0.0, 0.0)
EXPAND_ARGIDX_DTYPE_MACRO(float,   fp32, 0.0, 0.0)
EXPAND_ARGIDX_DTYPE_MACRO(double,  fp64, 0.0, 0.0)
EXPAND_ARGIDX_DTYPE_MACRO(int16_t, i16,  0,   0)
EXPAND_ARGIDX_DTYPE_MACRO(int,     i32,  0,   0)
EXPAND_ARGIDX_DTYPE_MACRO(int64_t, i64,  0,   0)
EXPAND_ARGIDX_DTYPE_MACRO(uint8_t, u8,   0,   0)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__
void __launch_bounds__(256) fn_r_max_bc_sub_exp_sum_bc_div__kernel(
    const float* __restrict__ var, /* 128x256x768 */
    float* __restrict__ var_6 /* 128x256x768 */
) {
  __builtin_assume(((int)blockIdx.x < 4096));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_0_rf_temp_buffer [ 1 ];
  float _var_0_temp_buffer [ 1 ];
  float _var_4_rf_temp_buffer [ 1 ];
  float _var_4_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_0 = _var_0_temp_buffer;
  float* var_0_rf = _var_0_rf_temp_buffer;
  float* var_0_rf__reduce_init = _var_0_rf_temp_buffer;
  float* var_4 = _var_4_temp_buffer;
  float* var_4_rf = _var_4_rf_temp_buffer;
  float* var_4_rf__reduce_init = _var_4_rf_temp_buffer;
  var_0_rf__reduce_init[0] = -3.40282347e+38f;
  for (int32_t k = 0; k < 24; k += 1) {
    float var_local = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 768) + (k * 32)) + (int)threadIdx.x)];
    var_0_rf[0] = max(var_0_rf[0], var_local);
  }
  var_0[0] = cinn_block_reduce_max_fp32(var_0_rf[0], shm32__fp32_reduce, true);
  var_4_rf__reduce_init[0] = 0.00000000f;
  for (int32_t k = 0; k < 24; k += 1) {
    float var_local = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 768) + (k * 32)) + (int)threadIdx.x)];
    var_4_rf[0] = (var_4_rf[0] + cinn_nvgpu_exp_fp32((var_local - var_0[0])));
  }
  var_4[0] = cinn_block_reduce_sum_fp32(var_4_rf[0], shm32__fp32_reduce, true);
  for (int32_t k = 0; k < 24; k += 1) {
    float var_local = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 768) + (k * 32)) + (int)threadIdx.x)];
    var_6[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 768) + (k * 32)) + (int)threadIdx.x)] = (cinn_nvgpu_exp_fp32((var_local - var_0[0])) / var_4[0]);
  }
}

__global__
void __launch_bounds__(256) fn_slice_slice_cast_scale_exp_full_add_full_div_mul_cast_mul_cast_bc_mul_reshape_abs_r_max_full_full_bc_bc_greater_equal_select_less_equal_select_reshape_scale_log2_ceil_bc_pow_bc_div_full_full_bc_bc_greater_equal_select_less_equal_select_cast_reshape__COND__FPA__FPA_trueAND_FPA_524288llGE1ll_BPA__BPA_AND_FPA_128llGE1ll_BPA__BPA___kernel(const bfloat16* __restrict__ var, const float* __restrict__ var_13, const float* __restrict__ var_31, float8e4m3* __restrict__ var_45)
{
  __builtin_assume(((int)blockIdx.x < 65536));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_16_temp_buffer [ 4 ];
  float _var_18_rf_temp_buffer [ 1 ];
  float _var_18_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16 = _var_16_temp_buffer;
  float* var_18 = _var_18_temp_buffer;
  float* var_18_rf = _var_18_rf_temp_buffer;
  float* var_18_rf__reduce_init = _var_18_rf_temp_buffer;
  for (int32_t k = 0; k < 4; k += 1) {
    bfloat16 var_local = var[((((((((int)blockIdx.x * 8) + (int)threadIdx.y) & 15) * 128) + (k * 32)) + (int)threadIdx.x) + (((((int)blockIdx.x * 8) + (int)threadIdx.y) / 16) * 4096))];
    bfloat16 var_local_0 = var[(((((((((int)blockIdx.x * 8) + (int)threadIdx.y) & 15) * 128) + (k * 32)) + (int)threadIdx.x) + (((((int)blockIdx.x * 8) + (int)threadIdx.y) / 16) * 4096)) + 2048)];
    float var_13_local = var_13[((((int)blockIdx.x * 8) + (int)threadIdx.y) / 16)];
    var_16[k] = (((float)((((bfloat16)((((float)(var_local)) * cinn_nvgpu_rcp_fp32((1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((float)(var_local))))))))) * var_local_0))) * var_13_local);
  };
  var_18_rf__reduce_init[0] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0 = 0; reduce_k_0_0 < 4; reduce_k_0_0 += 1) {
    var_18_rf[0] = max(var_18_rf[0], cinn_nvgpu_abs_fp32(var_16[reduce_k_0_0]));
  };
  var_18[0] = cinn_block_reduce_max_fp32(var_18_rf[0], shm32__fp32_reduce, true);
  for (int32_t j_0 = 0; j_0 < 4; j_0 += 1) {
    float var_31_local = var_31[0];
    // 提前计算的常量（如果可能）
    const float scale = 0.00223214296f;
    const float pow_val = cinn_nvgpu_pow_fp32(
      var_31_local,
      cinn_nvgpu_ceil_fp32(
        cinn_nvgpu_log2_fp32(
          scale * min(max(var_18[j_0], 9.99999975e-05f), 3.40282347e+38f)
        )
      )
    );
    // 主计算逻辑
    float tmp = var_16[j_0] / pow_val;
    float clipped_value = min(max(tmp, -448.0f), 448.0f);
    // 写入全局内存
    var_45[((((((((int)blockIdx.x * 8) + (int)threadIdx.y) & 15) * 128) + (j_0 * 32)) + (int)threadIdx.x) + (((((int)blockIdx.x * 8) + (int)threadIdx.y) / 16) * 2048))] = (float8e4m3)clipped_value; 
    // var_45[((((((((int)blockIdx.x * 8) + (int)threadIdx.y) & 15) * 128) + (j_0 * 32)) + (int)threadIdx.x) + (((((int)blockIdx.x * 8) + (int)threadIdx.y) / 16) * 2048))] = ((float8e4m3)(((((var_16[j_0] / cinn_nvgpu_pow_fp32(var_31_local, cinn_nvgpu_ceil_fp32(cinn_nvgpu_log2_fp32((0.00223214296f * (((var_18[j_0] <= 3.40282347e+38f)) ? (((var_18[j_0] >= 9.99999975e-05f)) ? var_18[j_0] : 9.99999975e-05f) : 3.40282347e+38f)))))) <= 448.000000f)) ? ((((var_16[j_0] / cinn_nvgpu_pow_fp32(var_31_local, cinn_nvgpu_ceil_fp32(cinn_nvgpu_log2_fp32((0.00223214296f * (((var_18[j_0] <= 3.40282347e+38f)) ? (((var_18[j_0] >= 9.99999975e-05f)) ? var_18[j_0] : 9.99999975e-05f) : 3.40282347e+38f)))))) >= -448.000000f)) ? (var_16[j_0] / cinn_nvgpu_pow_fp32(var_31_local, cinn_nvgpu_ceil_fp32(cinn_nvgpu_log2_fp32((0.00223214296f * (((var_18[j_0] <= 3.40282347e+38f)) ? (((var_18[j_0] >= 9.99999975e-05f)) ? var_18[j_0] : 9.99999975e-05f) : 3.40282347e+38f)))))) : -448.000000f) : 448.000000f)));
  };
}

int main() {
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if (!supportsCoopLaunch) {
    cerr << "cooperative launch is not supported" << endl;
    exit(1);
  }

  cerr << "initializing data" << endl;
  Tensor<curandState> gen(64*256, RandInit{2024});
  Tensor<int> sem(128);

  Tensor<bfloat16> var(32768*4096, Randn{gen});
  Tensor<float> var_13(32768*1, Randn{gen});
  Tensor<float> var_31(1, Two<float>{});
  Tensor<float8e4m3> ou0(32768*2048);

  cerr << "launching kernel" << endl;
  float avg_ms = GpuTimer().trial(100, [&](){
    // sem.clear();
    // fn_r_max_bc_sub_exp_sum_bc_div__kernel<<<4096, dim3(32, 8), sizeof(float)*32>>>(in0, ou0);
    fn_slice_slice_cast_scale_exp_full_add_full_div_mul_cast_mul_cast_bc_mul_reshape_abs_r_max_full_full_bc_bc_greater_equal_select_less_equal_select_reshape_scale_log2_ceil_bc_pow_bc_div_full_full_bc_bc_greater_equal_select_less_equal_select_cast_reshape__COND__FPA__FPA_trueAND_FPA_524288llGE1ll_BPA__BPA_AND_FPA_128llGE1ll_BPA__BPA___kernel<<<65536, dim3(32, 8), sizeof(float)*32>>>(var, var_13, var_31, ou0);

    // void* args[] = {&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &ou0, &ou1, &ou2, &ou3, &ou4, &ou5};
    // cudaLaunchCooperativeKernel(
    //     (void *)cinn_bn_fw, dim3(24, 13), dim3(32, 8), args, sizeof(float)*512, NULL);
  });
  cerr << "avg_us: " << avg_ms * 1e3 << endl;

  cerr << "validating" << endl;
  ou0.to_host();
}
