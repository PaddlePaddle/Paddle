// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#if !defined(_WIN32) && !defined(__APPLE__)
#include <cstddef>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

#define PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION 0
#define PADDLE_CUSTOM_RUNTIME_MINOR_VERSION 1
#define PADDLE_CUSTOM_RUNTIME_PATCH_VERSION 1

typedef enum {
  UNDEFINED = 0,
  BOOL,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  BFLOAT16,
} C_DataType;

typedef enum {
  ANY = 0,
  NHWC,
  NCHW,
  NCDHW,
  NDHWC,
  NUM_DATA_LAYOUTS,
  ALL_LAYOUT = ANY,
} C_DataLayout;

typedef enum {
  C_SUCCESS = 0,    // success
  C_WARNING,        // results may not meet expectation (such as an asynchronous
                    // interface is actually synchronous)
  C_FAILED,         // resource exhausted/query failed
  C_ERROR,          // invalid argument/wrong usage/uninitialized
  C_INTERNAL_ERROR  // plugin error
} C_Status;

typedef struct C_Device_st {
  int id;
} * C_Device;

typedef struct C_Stream_st* C_Stream;

typedef struct C_Event_st* C_Event;

typedef void (*C_Callback)(C_Device device,
                           C_Stream stream,
                           void* user_data,
                           C_Status* status);

typedef struct {
  size_t sz;
  void* data;
} C_CCLRootId;

typedef struct C_CCLComm_st* C_CCLComm;

typedef enum { SUM = 0, AVG, MAX, MIN, PRODUCT } C_CCLReduceOp;

typedef struct C_Profiler_st* C_Profiler;

void profiler_add_runtime_trace_event(C_Profiler prof, void* event);

void profiler_add_device_trace_event(C_Profiler prof, void* event);

struct C_DeviceInterface {
  // Core fill it and plugin must to check it
  size_t size;

  ///////////////////////
  // device manage api //
  ///////////////////////

  /**
   * @brief Initialize hardware
   *
   */
  C_Status (*initialize)();

  /**
   * @brief Deinitialize hardware
   *
   */
  C_Status (*finalize)();

  /**
   * @brief Initialize device
   *
   * @param[C_Device] device     Core fill it with a logical id, and then plugin
   * must replace it with a physical id
   */
  C_Status (*init_device)(const C_Device device);

  /**
   * @brief Set current device
   *
   * @param[C_Device] device     Core fill it with a physical id
   */
  C_Status (*set_device)(const C_Device device);

  /**
   * @brief Get current device
   *
   * @param[C_Device] device     Plugin fill it with a physical id
   */
  C_Status (*get_device)(const C_Device device);

  /**
   * @brief Deinitialize device
   *
   * @param[C_Device] device     Core fill it with a physical id
   */
  C_Status (*deinit_device)(const C_Device device);

  /**
   * @brief Create a stream
   *
   * @param[C_Device] device     Core fill it with a physical id
   * @param[C_Stream*] stream    Plugin create a stream and fill it
   */
  C_Status (*create_stream)(const C_Device device, C_Stream* stream);

  /**
   * @brief Destroy a stream
   *
   * @param[C_Device] device     Core fill it with a physical id
   * @param[C_Stream] stream
   */
  C_Status (*destroy_stream)(const C_Device device, C_Stream stream);

  /**
   * @brief Query a stream
   *
   * @param[C_Device] device     Core fill it with a physical id
   * @param[C_Stream] stream
   */
  C_Status (*query_stream)(const C_Device device, C_Stream stream);

  /**
   * @brief Add a callback to stream
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[C_Callback] callback
   * @param[void*]      user_data
   */
  C_Status (*stream_add_callback)(const C_Device device,
                                  C_Stream stream,
                                  C_Callback callback,
                                  void* user_data);

  /**
   * @brief Create an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Event*]   event      Plugin create an event and fill it
   */
  C_Status (*create_event)(const C_Device device, C_Event* event);

  /**
   * @brief Record an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[C_Event]    event
   */
  C_Status (*record_event)(const C_Device device,
                           C_Stream stream,
                           C_Event event);

  /**
   * @brief Destroy an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Event]    event
   */
  C_Status (*destroy_event)(const C_Device device, C_Event event);

  /**
   * @brief Query an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Event]    event
   */
  C_Status (*query_event)(const C_Device device, C_Event event);

  /**
   * @brief Synchronize a device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   */
  C_Status (*synchronize_device)(const C_Device device);

  /**
   * @brief Synchronize a stream
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   */
  C_Status (*synchronize_stream)(const C_Device device, C_Stream stream);

  /**
   * @brief Synchronize an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Event]    event
   */
  C_Status (*synchronize_event)(const C_Device device, C_Event event);

  /**
   * @brief Make a stream wait on an event
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[C_Event]    event
   */
  C_Status (*stream_wait_event)(const C_Device device,
                                C_Stream stream,
                                C_Event event);

  void* reserved_dev_api[8];

  ///////////////////////
  // memory manage api //
  ///////////////////////

  /**
   * @brief Device memory allocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void**]     ptr        Plugin allocate an address and fill it
   * @param[size_t]     size
   */
  C_Status (*device_memory_allocate)(const C_Device device,
                                     void** ptr,
                                     size_t size);

  /**
   * @brief Device memory deallocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      ptr
   * @param[size_t]     size
   */
  C_Status (*device_memory_deallocate)(const C_Device device,
                                       void* ptr,
                                       size_t size);

  /**
   * @brief Device memory set
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      ptr
   * @param[unsigned char] value
   * @param[size_t]     size
   */
  C_Status (*device_memory_set)(const C_Device device,
                                void* ptr,
                                unsigned char value,
                                size_t size);

  /**
   * @brief Host memory allocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void**]     ptr        Plugin allocate an address and fill it
   * @param[size_t]     size
   */
  C_Status (*host_memory_allocate)(const C_Device device,
                                   void** ptr,
                                   size_t size);

  /**
   * @brief Host memory deallocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      ptr
   * @param[size_t]     size
   */
  C_Status (*host_memory_deallocate)(const C_Device device,
                                     void* ptr,
                                     size_t size);

  /**
   * @brief Unified memory allocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void**]     ptr        Plugin allocate an address and fill it
   * @param[size_t]     size
   */
  C_Status (*unified_memory_allocate)(const C_Device device,
                                      void** ptr,
                                      size_t size);

  /**
   * @brief Unified memory deallocate
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      ptr
   * @param[size_t]     size
   */
  C_Status (*unified_memory_deallocate)(const C_Device device,
                                        void* ptr,
                                        size_t size);

  /**
   * @brief Memory copy from host to device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*memory_copy_h2d)(const C_Device device,
                              void* dst,
                              const void* src,
                              size_t size);

  /**
   * @brief Memory copy from device to host
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*memory_copy_d2h)(const C_Device device,
                              void* dst,
                              const void* src,
                              size_t size);

  /**
   * @brief Memory copy from device to device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*memory_copy_d2d)(const C_Device device,
                              void* dst,
                              const void* src,
                              size_t size);

  /**
   * @brief Peer memory copy from device to device
   *
   * @param[C_Device]   dst_device     Core fill it with a physical id
   * @param[C_Device]   src_device     Core fill it with a physical id
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*memory_copy_p2p)(const C_Device dst_device,
                              const C_Device src_device,
                              void* dst,
                              const void* src,
                              size_t size);

  /**
   * @brief Asynchonrize memory copy from host to device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*async_memory_copy_h2d)(const C_Device device,
                                    C_Stream stream,
                                    void* dst,
                                    const void* src,
                                    size_t size);

  /**
   * @brief Asynchonrize memory copy from device to host
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*async_memory_copy_d2h)(const C_Device device,
                                    C_Stream stream,
                                    void* dst,
                                    const void* src,
                                    size_t size);

  /**
   * @brief Asynchonrize memory copy from device to device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*async_memory_copy_d2d)(const C_Device device,
                                    C_Stream stream,
                                    void* dst,
                                    const void* src,
                                    size_t size);

  /**
   * @brief Peer asynchonrize memory copy from host to device
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[C_Stream]   stream
   * @param[void*]      dst
   * @param[void*]      src
   * @param[size_t]     size
   */
  C_Status (*async_memory_copy_p2p)(const C_Device dst_device,
                                    const C_Device src_device,
                                    C_Stream stream,
                                    void* dst,
                                    const void* src,
                                    size_t size);

  void* reserved_mem_api[8];

  //////////////
  // info api //
  //////////////

  /**
   * @brief Get visible device count
   *
   * @param[size_t*]    count       Plugin fill it
   */
  C_Status (*get_device_count)(size_t* count);

  /**
   * @brief Get visible device list
   *
   * @param[size_t*]    devices     Plugin fill it
   */
  C_Status (*get_device_list)(size_t* devices);

  /**
   * @brief Device memory statistic
   *
   * @param[C_Device]   device     Core fill it with a physical id
   * @param[size_t*]    total_memory
   * @param[size_t*]    free_memory
   * @param[size_t*]    used_memory
   */
  C_Status (*device_memory_stats)(const C_Device device,
                                  size_t* total_memory,
                                  size_t* free_memory);

  /**
   * @brief Device minimum chunk size
   *
   * @param[size_t*]    count
   */
  C_Status (*device_min_chunk_size)(const C_Device device, size_t* count);

  /**
   * @brief Device maximum chunk size
   *
   * @param[size_t*]    count
   */
  C_Status (*device_max_chunk_size)(const C_Device device, size_t* count);

  /**
   * @brief Device maximum alloc size
   *
   * @param[size_t*]    count
   */
  C_Status (*device_max_alloc_size)(const C_Device device, size_t* count);

  /**
   * @brief Device extra padding size
   *
   * @param[size_t*]    size
   */
  C_Status (*device_extra_padding_size)(const C_Device device, size_t* size);

  /**
   * @brief Device initial allocated size
   *
   * @param[size_t*]    size
   */
  C_Status (*device_init_alloc_size)(const C_Device device, size_t* size);

  /**
   * @brief Device reallocated size
   *
   * @param[size_t*]    size
   */
  C_Status (*device_realloc_size)(const C_Device device, size_t* size);

  /**
   * @brief Get compute capability
   *
   * @param[size_t*]    compute_capability
   */
  C_Status (*get_compute_capability)(size_t* compute_capability);

  /**
   * @brief Get runtime version
   *
   * @param[size_t*]    version
   */
  C_Status (*get_runtime_version)(size_t* version);

  /**
   * @brief Get driver version
   *
   * @param[size_t*]    version
   */
  C_Status (*get_driver_version)(size_t* version);

  void* reserved_info_api[8];

  //////////////
  // ccl api //
  //////////////

  /**
   * @brief Get size of unique id
   *
   * @param[size_t*]         size
   */
  C_Status (*xccl_get_unique_id_size)(size_t* size);

  /**
   * @brief Get unique id
   *
   * @param[C_CCLRootId*]    unique_id
   */
  C_Status (*xccl_get_unique_id)(C_CCLRootId* unique_id);

  /**
   * @brief Initialize communicator
   *
   * @param[size_t]          ranks
   * @param[C_CCLRootId*]    unique_id
   * @param[size_t]          rank
   * @param[C_CCLComm*]      comm
   */
  C_Status (*xccl_comm_init_rank)(size_t ranks,
                                  C_CCLRootId* unique_id,
                                  size_t rank,
                                  C_CCLComm* comm);

  /**
   * @brief Destroy communicator
   *
   * @param[C_CCLComm]  comm
   */
  C_Status (*xccl_destroy_comm)(C_CCLComm comm);

  C_Status (*xccl_all_reduce)(void* send_buf,
                              void* recv_buf,
                              size_t count,
                              C_DataType data_type,
                              C_CCLReduceOp op,
                              C_CCLComm comm,
                              C_Stream stream);

  C_Status (*xccl_broadcast)(void* buf,
                             size_t count,
                             C_DataType data_type,
                             size_t root,
                             C_CCLComm comm,
                             C_Stream stream);

  C_Status (*xccl_reduce)(void* send_buf,
                          void* recv_buf,
                          size_t count,
                          C_DataType data_type,
                          C_CCLReduceOp op,
                          size_t root,
                          C_CCLComm comm,
                          C_Stream stream);

  C_Status (*xccl_all_gather)(void* send_buf,
                              void* recv_buf,
                              size_t count,
                              C_DataType data_type,
                              C_CCLComm comm,
                              C_Stream stream);

  C_Status (*xccl_reduce_scatter)(void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  C_DataType data_type,
                                  C_CCLReduceOp op,
                                  C_CCLComm comm,
                                  C_Stream stream);

  C_Status (*xccl_group_start)();

  C_Status (*xccl_group_end)();

  C_Status (*xccl_send)(void* send_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t dest_rank,
                        C_CCLComm comm,
                        C_Stream stream);

  C_Status (*xccl_recv)(void* recv_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t src_rank,
                        C_CCLComm comm,
                        C_Stream stream);

  void* reserved_ccl_api[8];

  //////////////////
  // profiler api //
  //////////////////

  C_Status (*profiler_initialize)(C_Profiler prof, void** user_data);

  C_Status (*profiler_finalize)(C_Profiler prof, void* user_data);

  C_Status (*profiler_prepare_tracing)(C_Profiler prof, void* user_data);

  C_Status (*profiler_start_tracing)(C_Profiler prof, void* user_data);

  C_Status (*profiler_stop_tracing)(C_Profiler prof, void* user_data);

  C_Status (*profiler_collect_trace_data)(C_Profiler prof,
                                          uint64_t start_ns,
                                          void* user_data);

  void* reserved_profiler_api[8];

  ///////////////
  // other api //
  ///////////////

  /**
   * @brief y = alpha * x + beta * y
   *
   */
  C_Status (*blas_axpby)(const C_Device device,
                         C_Stream stream,
                         C_DataType dtype,
                         size_t numel,
                         float alpha,
                         void* x,
                         float beta,
                         void* y);
  void* reserved_other_api[7];
};

struct CustomRuntimeVersion {
  size_t major, minor, patch;
};

struct CustomRuntimeParams {
  // Core fill it and plugin must to check it
  size_t size;
  // Plugin fill it
  C_DeviceInterface* interface;
  // Plugin fill it and Core will to check it
  CustomRuntimeVersion version;
  // Plugin fill it
  char* device_type;
  // Plugin fill it
  char* sub_device_type;

  char reserved[32];
};

#define PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params)              \
  if ((params)->size != sizeof(CustomRuntimeParams) &&           \
      (params)->interface->size != sizeof(C_DeviceInterface)) {  \
    return;                                                      \
  }                                                              \
  (params)->version.major = PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION; \
  (params)->version.minor = PADDLE_CUSTOM_RUNTIME_MINOR_VERSION; \
  (params)->version.patch = PADDLE_CUSTOM_RUNTIME_PATCH_VERSION;

// Plugin implement it and fill CustomRuntimeParams
void InitPlugin(CustomRuntimeParams*);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
