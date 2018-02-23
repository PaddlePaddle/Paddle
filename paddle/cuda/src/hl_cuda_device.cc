/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// clang-format off
// Because clang-format 4.X and clang-format 3.8+ format
// following lines in different. So disable clang-format.
#include "hl_cuda.h"
#include <cuda_profiler_api.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include "hl_cuda.ph"
#include "hl_thread.ph"
#include "paddle/utils/Logging.h"
#include "paddle/utils/DynamicLoader.h"
// clang-format on

namespace dynload {

std::once_flag curand_dso_flag;
void *curand_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load curand routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CURAND_WRAP(__name)                                       \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    curandStatus_t operator()(Args... args) {                                  \
      typedef curandStatus_t (*curandFunc)(Args...);                           \
      std::call_once(curand_dso_flag, GetCurandDsoHandle, &curand_dso_handle); \
      void *p_##__name = dlsym(curand_dso_handle, #__name);                    \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);                \
    }                                                                          \
  } __name; /* struct DynLoad__##__name */
#else
#define DYNAMIC_LOAD_CURAND_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    curandStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name; /* struct DynLoad__##__name */
#endif

/* include all needed curand functions in HPPL */
// clang-format off
#define CURAND_RAND_ROUTINE_EACH(__macro)    \
  __macro(curandCreateGenerator)             \
  __macro(curandSetStream)                   \
  __macro(curandSetPseudoRandomGeneratorSeed)\
  __macro(curandGenerateUniform)             \
  __macro(curandGenerateUniformDouble)
// clang-format on

CURAND_RAND_ROUTINE_EACH(DYNAMIC_LOAD_CURAND_WRAP)

#undef CURAND_RAND_ROUTINE_EACH
#undef DYNAMIC_LOAD_CURAND_WRAP

} /* namespace dynload */

/**
 * @brief   global resource.
 */
int g_system_device_num = 0;                /* system device number */
int device_num = 0;                         /* use    device number */
hl_device_prop *g_device;                   /* device info table */
__thread thread_device_resources *t_device; /* device resources table */
int g_cuda_lib_version = 0;

/* number of global stream */
#define NUMBER_OF_GLOBAL_STREAM (HPPL_THREAD_STREAM_1)
/* number of thread stream */
#define NUMBER_OF_THREAD_STREAM (HPPL_STREAM_END - HPPL_THREAD_STREAM_1)
/* sizeof of device memory */
#define HPPL_GPU_MEMORY_SIZE (256 * 4)

/**
 * Check build-in cuda function using glog and it **does not**
 * support << operator for more details error info.
 */
#define CHECK_CUDA(cudaFunc)                                         \
  do {                                                               \
    cudaError_t cudaStat = cudaFunc;                                 \
    CHECK_EQ(cudaSuccess, cudaStat) << "Cuda Error: "                \
                                    << cudaGetErrorString(cudaStat); \
  } while (0)

/**
 * @brief   thread resource.
 */
__thread _hl_thread_resource t_resource = {{0},    /* stream */
                                           0,      /* handle */
                                           0,      /* gen */
                                           0,      /* cudnn_handle */
                                           0,      /* cudnn_desc */
                                           NULL,   /* gen_mutex */
                                           NULL,   /* gpu_mem */
                                           NULL,   /* cpu_mem */
                                           0,      /* event */
                                           -1,     /* device */
                                           0,      /* major */
                                           false}; /* is_init */

__thread cudaStream_t default_stream = 0;
__thread bool g_sync_flag = true;
bool hl_start_flag = false;

inline pid_t gettid() {
#if defined(__APPLE__) || defined(__OSX__)
  // syscall is deprecated: first deprecated in macOS 10.12.
  // syscall is unsupported;
  // syscall pid_t tid = syscall(SYS_thread_selfid);
  uint64_t tid;
  pthread_threadid_np(NULL, &tid);
#else
#ifndef __NR_gettid
#define __NR_gettid 224
#endif
  pid_t tid = syscall(__NR_gettid);
#endif
  CHECK_NE((int)tid, -1);
  return tid;
}

void hl_init(int device) {
  CHECK(hl_start_flag) << "[Init failed] hl_start() did not succeed.";

  /* thread has been initialized */
  if (true == t_resource.is_init) {
    hl_set_device(device);
    return;
  }

  /* create thread devcie resources */
  char *tmp;
  thread_device_resources device_res;
  tmp = (char *)malloc(g_system_device_num * sizeof(thread_device_resources *) +
                       device_num * sizeof(_thread_device_resources));
  CHECK_NOTNULL(tmp);
  t_device = (thread_device_resources *)tmp;
  device_res = (thread_device_resources)(
      (char *)tmp + g_system_device_num * sizeof(thread_device_resources *));
  memset(t_device, 0, g_system_device_num * sizeof(thread_device_resources *));

  char *tmp_stream = (char *)malloc(device_num * NUMBER_OF_THREAD_STREAM *
                                    sizeof(cudaStream_t));
  CHECK_NOTNULL(tmp_stream);

  int num = 0;
  for (int dev = 0; dev < g_system_device_num; dev++) {
    if (!g_device[dev]) {
      continue;
    }

    t_device[dev] = &device_res[num];
    t_device[dev]->stream =
        (cudaStream_t *)(tmp_stream +
                         num * NUMBER_OF_THREAD_STREAM * sizeof(cudaStream_t));

    hl_create_thread_resources(dev, t_device[dev]);
    num++;
  }

  hl_cudnn_desc_init(&t_resource.cudnn_desc);

  /* thread initialization is complete */
  t_resource.is_init = true;
  /* set device */
  t_resource.device = -1;
  hl_set_device(device);
}

void hl_fini() {
  if (false == t_resource.is_init) {
    return;
  }

  /* hppl stream fini */
  t_resource.device = -1;
  for (int i = NUMBER_OF_GLOBAL_STREAM; i < HPPL_STREAM_END; i++) {
    t_resource.stream[i] = 0;
  }

  char *tmp = (char *)t_device;
  char *tmp_stream = NULL;
  for (int dev = 0; dev < g_system_device_num; dev++) {
    if (!t_device[dev]) {
      continue;
    }
    if (!tmp_stream) {
      tmp_stream = (char *)t_device[dev]->stream;
    }
    for (int j = 0; j < NUMBER_OF_THREAD_STREAM; j++) {
      CHECK_CUDA(cudaStreamDestroy(t_device[dev]->stream[j]));
    }

    /* free device memory */
    hl_free_mem_device(t_device[dev]->gpu_mem);
    hl_free_mem_host(t_device[dev]->cpu_mem);
    CHECK_CUDA(cudaEventDestroy(t_device[dev]->mem_event));
  }

  free(tmp);
  free(tmp_stream);
  t_resource.is_init = false;
}

int hl_get_device_count() { return device_num; }

void hl_set_device(int device) {
  if (device == t_resource.device) {
    return;
  }

  CHECK(device >= 0 && device < g_system_device_num && g_device[device])
      << "Device: " << device << " is not specified in startup.";

  CHECK_CUDA(cudaSetDevice(device));

  /* switch thread stream */
  for (int i = 0; i < NUMBER_OF_GLOBAL_STREAM; i++) {
    t_resource.stream[i] = g_device[device]->device_resources->stream[i];
  }

  if (true == t_resource.is_init) {
    for (int i = NUMBER_OF_GLOBAL_STREAM; i < HPPL_STREAM_END; i++) {
      t_resource.stream[i] =
          t_device[device]->stream[i - NUMBER_OF_GLOBAL_STREAM];
    }
    t_resource.gpu_mem = t_device[device]->gpu_mem;
    t_resource.cpu_mem = t_device[device]->cpu_mem;
    t_resource.event = t_device[device]->mem_event;
  }

  t_resource.handle = g_device[device]->device_resources->handle;
  t_resource.gen = g_device[device]->device_resources->gen;
  t_resource.cudnn_handle = g_device[device]->device_resources->cudnn_handle;
  t_resource.gen_mutex = g_device[device]->device_resources->gen_mutex;
  t_resource.device = device;
  t_resource.major = g_device[device]->major;
  default_stream = t_resource.stream[0];
}

int hl_get_device() {
  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  return device;
}

void *hl_malloc_device(size_t size) {
  void *dest_d;

  CHECK(size) << __func__ << ": the size for device memory is 0, please check.";
  CHECK_CUDA(cudaMalloc((void **)&dest_d, size));

  return dest_d;
}

void hl_free_mem_device(void *dest_d) {
  CHECK_NOTNULL(dest_d);

  cudaError_t err = cudaFree(dest_d);
  CHECK(cudaSuccess == err || cudaErrorCudartUnloading == err)
      << hl_get_device_error_string();
}

void *hl_malloc_host(size_t size) {
  void *dest_h;

  CHECK(size) << __func__ << ": the size for device memory is 0, please check.";
  CHECK_CUDA(cudaHostAlloc((void **)&dest_h, size, cudaHostAllocDefault));

  return dest_h;
}

void hl_free_mem_host(void *dest_h) {
  CHECK_NOTNULL(dest_h);

  cudaError_t err = cudaFreeHost(dest_h);
  CHECK(cudaSuccess == err || cudaErrorCudartUnloading == err)
      << hl_get_device_error_string();
}

void hl_memcpy(void *dst, void *src, size_t size) {
  if (0 == size) {
    return;
  }
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}

void hl_memset_device(void *dest_d, int value, size_t size) {
  CHECK_CUDA(cudaMemset(dest_d, value, size));
}

void hl_memcpy_host2device(void *dest_d, void *src_h, size_t size) {
  if (0 == size) {
    return;
  }
  CHECK_NOTNULL(src_h);
  CHECK_NOTNULL(dest_d);
  CHECK_CUDA(cudaMemcpy(dest_d, src_h, size, cudaMemcpyHostToDevice));
}

void hl_memcpy_device2host(void *dest_h, void *src_d, size_t size) {
  if (0 == size) {
    return;
  }
  CHECK_NOTNULL(dest_h);
  CHECK_NOTNULL(src_d);
  CHECK_CUDA(cudaMemcpy(dest_h, src_d, size, cudaMemcpyDeviceToHost));
}

void hl_memcpy_device2device(void *dest_d, void *src_d, size_t size) {
  if (0 == size) {
    return;
  }
  CHECK_NOTNULL(dest_d);
  CHECK_NOTNULL(src_d);
  CHECK_CUDA(cudaMemcpy(dest_d, src_d, size, cudaMemcpyDeviceToDevice));
}

void hl_memcpy_async(void *dst, void *src, size_t size, hl_stream_t stream) {
  cudaStream_t cu_stream;

  if (0 == size) {
    return;
  }
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_LT(stream, HPPL_STREAM_END);
  cu_stream = t_resource.stream[stream];

  CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, cu_stream));
}

void hl_start() {
  hl_specify_devices_start(NULL, 0);
  /* set default device */
  hl_set_device(0);
}

bool hl_device_can_access_peer(int device, int peerDevice) {
  int canAccessPeer;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice));

  if (canAccessPeer == 1) {
    return true;
  } else {
    return false;
  }
}

void hl_device_enable_peer_access(int peerDevice) {
  cudaError_t err = cudaDeviceEnablePeerAccess(peerDevice, 0);
  if (cudaErrorPeerAccessAlreadyEnabled == err) {
    cudaGetLastError();
  } else {
    CHECK_CUDA(err);
  }
}

void hl_create_global_resources(hl_device_prop device_prop) {
  struct cudaDeviceProp cu_prop;
  int device = device_prop->device;
  global_device_resources device_res = device_prop->device_resources;

  CHECK_CUDA(cudaSetDevice(device));
  /* device properties */
  CHECK_CUDA(cudaGetDeviceProperties(&cu_prop, device));

  device_prop->major = cu_prop.major;
  device_prop->minor = cu_prop.minor;
  strncpy(device_prop->device_name, cu_prop.name, 256);
  device_prop->device_mem = cu_prop.totalGlobalMem;

  /* create device stream */
  for (int j = 0; j < NUMBER_OF_GLOBAL_STREAM; j++) {
    CHECK_CUDA(cudaStreamCreate(&device_res->stream[j]));
  }

  /* cublas init */
  hl_cublas_init(&device_res->handle, device_res->stream[0]);

  /* create curand gen */
  CHECK_EQ(dynload::curandCreateGenerator(&device_res->gen,
                                          CURAND_RNG_PSEUDO_DEFAULT),
           CURAND_STATUS_SUCCESS)
      << "[Start failed] Curand init failed.";

  CHECK_EQ(dynload::curandSetStream(device_res->gen, device_res->stream[0]),
           CURAND_STATUS_SUCCESS)
      << "[Start failed] Curand set stream failed!";

  /* create cudnn handle */
  hl_cudnn_init(&device_res->cudnn_handle, device_res->stream[0]);

  int seed = gettid();
  CHECK_EQ(dynload::curandSetPseudoRandomGeneratorSeed(device_res->gen,
                                                       seed + device),
           CURAND_STATUS_SUCCESS);

  device_res->gen_mutex = (pthread_mutex_t *)(malloc(sizeof(pthread_mutex_t)));
  pthread_mutex_init(device_res->gen_mutex, NULL);

  CHECK_CUDA(cudaRuntimeGetVersion(&g_cuda_lib_version));
}

int hl_get_cuda_version() { return g_cuda_lib_version; }

void hl_create_thread_resources(int device,
                                thread_device_resources device_res) {
  CHECK_CUDA(cudaSetDevice(device));

  /* create thread stream */
  for (int j = 0; j < NUMBER_OF_THREAD_STREAM; j++) {
    CHECK_CUDA(cudaStreamCreate(&device_res->stream[j]));
  }

  /* allocation device memory */
  device_res->gpu_mem = (real *)hl_malloc_device(HPPL_GPU_MEMORY_SIZE);

  /* allocation host memory */
  device_res->cpu_mem = (real *)hl_malloc_host(HPPL_GPU_MEMORY_SIZE);

  CHECK_CUDA(cudaEventCreate(&device_res->mem_event));
}

void hl_specify_devices_start(int *device, int number) {
  if (hl_start_flag) return;

  /* 1. get the number of devices */
  CHECK_CUDA(cudaGetDeviceCount(&g_system_device_num));
  CHECK_NE(g_system_device_num, 0) << "[Start failed] there is no GPU device";
  if (device == NULL) {
    number = g_system_device_num;
  }

  /* 2. check device & create device property table */
  CHECK_LE(number, g_system_device_num)
      << "[Start failed] System does not have enough device. "
      << "Device number: " << g_system_device_num << "Input number: " << number;

  char *tmp;
  hl_device_prop device_prop;
  tmp = (char *)malloc(g_system_device_num * sizeof(hl_device_prop *) +
                       number * sizeof(_hl_device_prop));
  CHECK(tmp) << "[Start failed] System memory is not enough.";

  g_device = (hl_device_prop *)tmp;
  device_prop = (hl_device_prop)(
      (char *)tmp + g_system_device_num * sizeof(hl_device_prop *));
  memset(g_device, 0, g_system_device_num * sizeof(hl_device_prop *));
  int num = 0;
  for (int i = 0; i < number; i++) {
    int dev;
    if (device == NULL) {
      dev = i;
    } else {
      dev = device[i];
    }

    CHECK_LT(dev, g_system_device_num)
        << "[Start failed] The specified device number is "
        << "out of range. Max device number: " << g_system_device_num - 1
        << " Specified devcie number: " << dev;

    if (g_device[dev]) {
      /* Warning */
      LOG(WARNING) << "[Warning] Repeat specify device: " << dev;
      continue;
    }

    g_device[dev] = &device_prop[num];
    g_device[dev]->device = dev;
    num++;
  }
  device_num = num;

  /* 3.  create global device resources */
  char *tmp_res = (char *)malloc(device_num * sizeof(_global_device_resources));
  CHECK_NOTNULL(tmp_res);

  char *tmp_stream = (char *)malloc(device_num * NUMBER_OF_GLOBAL_STREAM *
                                    sizeof(cudaStream_t));
  CHECK_NOTNULL(tmp_stream);

  num = 0;
  for (int i = 0; i < g_system_device_num; i++) {
    if (!g_device[i]) {
      continue;
    }

    g_device[i]->device_resources = (global_device_resources)(
        tmp_res + num * sizeof(_global_device_resources));
    g_device[i]->device_resources->stream =
        (cudaStream_t *)(tmp_stream +
                         num * NUMBER_OF_GLOBAL_STREAM * sizeof(cudaStream_t));

    hl_create_global_resources(g_device[i]);
    num++;
  }

  /* hl_start() is ok */
  hl_start_flag = true;
  /* set default device */
  if (device == NULL) {
    hl_set_device(0);
  } else {
    hl_set_device(device[0]);
  }
}

void hl_rand(real *dest_d, size_t num) {
  pthread_mutex_lock(t_resource.gen_mutex);
  CHECK_EQ(
#ifndef PADDLE_TYPE_DOUBLE
      dynload::curandGenerateUniform(t_resource.gen, dest_d, num),
#else
      dynload::curandGenerateUniformDouble(t_resource.gen, dest_d, num),
#endif
      CURAND_STATUS_SUCCESS);
  pthread_mutex_unlock(t_resource.gen_mutex);
  CHECK_SYNC("hl_rand failed");
}

void hl_srand(unsigned int seed) {
  pthread_mutex_lock(t_resource.gen_mutex);
  CHECK_EQ(dynload::curandSetPseudoRandomGeneratorSeed(t_resource.gen, seed),
           CURAND_STATUS_SUCCESS);
  pthread_mutex_unlock(t_resource.gen_mutex);
}

void hl_set_sync_flag(bool flag) { g_sync_flag = flag; }

bool hl_get_sync_flag() { return g_sync_flag; }

void hl_stream_synchronize(hl_stream_t stream) {
  cudaStream_t cu_stream;

  CHECK_LT(stream, HPPL_STREAM_END) << __func__
                                    << ": the parameter stream is error.";

  cu_stream = t_resource.stream[stream];
  CHECK_CUDA(cudaStreamSynchronize(cu_stream));
}

void hl_create_event(hl_event_t *event) {
  CHECK_NOTNULL(event);

  struct _hl_event_st *st_event =
      (struct _hl_event_st *)malloc(sizeof(struct _hl_event_st));

  CHECK_CUDA(cudaEventCreate(&st_event->cu_event));

  *event = st_event;
}

float hl_event_elapsed_time(hl_event_t start, hl_event_t end) {
  float time;
  CHECK_NOTNULL(start);
  CHECK_NOTNULL(end);

  CHECK_CUDA(cudaEventElapsedTime(&time, start->cu_event, end->cu_event));
  return time;
}

void hl_stream_record_event(hl_stream_t stream, hl_event_t event) {
  cudaStream_t cu_stream;

  CHECK_NOTNULL(event);
  CHECK_LT(stream, HPPL_STREAM_END) << __func__
                                    << ": the parameter stream is error.";

  cu_stream = t_resource.stream[stream];
  CHECK_CUDA(cudaEventRecord(event->cu_event, cu_stream));
}

void hl_stream_wait_event(hl_stream_t stream, hl_event_t event) {
  cudaStream_t cu_stream;

  CHECK_NOTNULL(event);
  CHECK_LT(stream, HPPL_STREAM_END) << __func__
                                    << ": the parameter stream is error.";

  cu_stream = t_resource.stream[stream];
  CHECK_CUDA(cudaStreamWaitEvent(cu_stream, event->cu_event, 0));
}

void hl_destroy_event(hl_event_t event) {
  CHECK_NOTNULL(event);
  CHECK_CUDA(cudaEventDestroy(event->cu_event));

  free(event);
  event = NULL;
}

void hl_event_synchronize(hl_event_t event) {
  CHECK_NOTNULL(event);
  CHECK_CUDA(cudaEventSynchronize(event->cu_event));
}

void hl_get_device_name(char *name, int len, int device) {
  CHECK_NOTNULL(name);
  CHECK(device >= 0 && device < g_system_device_num && g_device[device])
      << "Device(" << device << ") is not specified in startup.";

  strncpy(name, g_device[device]->device_name, len);
}

void hl_get_device_memory(size_t *mem_size, int device) {
  CHECK_NOTNULL(mem_size);
  CHECK(device >= 0 && device < g_system_device_num && g_device[device])
      << "Device(" << device << ") is not specified in startup.";

  *mem_size = g_device[device]->device_mem;
}

void hl_get_device_compute_capability(int *major, int *minor, int device) {
  CHECK_NOTNULL(major);
  CHECK_NOTNULL(minor);
  CHECK(device >= 0 && device < g_system_device_num && g_device[device])
      << "Device(" << device << ") is not specified in startup.";

  *major = g_device[device]->major;
  *minor = g_device[device]->minor;
}

int hl_get_device_last_error() { return (int)cudaGetLastError(); }

const char *hl_get_device_error_string() {
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}

const char *hl_get_device_error_string(size_t err) {
  return cudaGetErrorString((cudaError_t)err);
}

void hl_device_synchronize() { CHECK_CUDA(cudaDeviceSynchronize()); }
void hl_set_device_flags_block() {
  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
}

bool hl_cuda_event_is_ready(hl_event_t event) {
  cudaError_t err = cudaEventQuery(event->cu_event);
  CHECK(cudaSuccess == err || cudaErrorNotReady == err);

  if (cudaErrorNotReady == err) {
    return false;
  }
  return true;
}

void hl_profiler_start() { CHECK_CUDA(cudaProfilerStart()); }

void hl_profiler_end() { CHECK_CUDA(cudaProfilerStop()); }
