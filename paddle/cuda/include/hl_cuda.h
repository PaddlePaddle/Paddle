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

#ifndef HL_CUDA_H_
#define HL_CUDA_H_

#include <string>
#include "hl_base.h"

/**
 * @brief   HPPL event.
 */
typedef struct _hl_event_st *hl_event_t;

/**
 * @brief return cuda runtime api version.
 */
extern int hl_get_cuda_lib_version();

/**
 * @brief   HPPL strat(Initialize all GPU).
 */
extern void hl_start();

/**
 * @brief   HPPL start(Initialize the specific GPU).
 *
 * @param[in]   device  device id(0, 1......).
 *                      if device is NULL, will start all GPU.
 * @param[in]   number  number of devices.
 */
extern void hl_specify_devices_start(int *device, int number);

/**
 * @brief   Queries if a device may directly access a peer device's memory.
 *
 * @param[in]   device      Device from which allocations on peerDevice are
 *                          to be directly accessed.
 * @param[in]   peerDevice  Device on which the allocations to be directly
 *                          accessed by device reside.
 *
 * @return  Returns true if device is capable of directly accessing memory
 *          from peerDevice and false otherwise.
 */
bool hl_device_can_access_peer(int device, int peerDevice);

/**
 * @brief   Enables direct access to memory allocations on a peer device.
 *
 * @param[in]   peerDevice  Peer device to enable direct access to from the
 *                          current device
 */
void hl_device_enable_peer_access(int peerDevice);

/**
 * @brief   Init a work thread.
 *
 * @param[in]   device  device id.
 */
extern void hl_init(int device);

/**
 * @brief   Finish a work thread.
 */
extern void hl_fini();

/**
 * @brief   Set synchronous/asynchronous flag.
 *
 * @param[in]   flag    true(default), set synchronous flag.
 *                      false, set asynchronous flag.
 *
 *
 * @note    This setting is only valid for the current worker thread.
 */
extern void hl_set_sync_flag(bool flag);

/**
 * @brief   Get synchronous/asynchronous flag.
 *
 * @return  Synchronous call true.
 *          Asynchronous call false.
 *
 */
extern bool hl_get_sync_flag();

/**
 * @brief   Returns the number of compute-capable devices.
 *
 */
extern int hl_get_device_count();

/**
 * @brief   Set device to be used.
 *
 * @param[in]   device  device id.
 *
 */
extern void hl_set_device(int device);

/**
 * @brief   Returns which device is currently being used.
 *
 * @return  device  device id.
 *
 */
extern int hl_get_device();

/**
 * @brief   Allocate device memory.
 *
 * @param[in]   size     size in bytes to copy.
 *
 * @return      dest_d   pointer to device memory.
 */
extern void *hl_malloc_device(size_t size);

/**
 * @brief   Free device memory.
 *
 * @param[in]   dest_d  pointer to device memory.
 *
 */
extern void hl_free_mem_device(void *dest_d);

/**
 * @brief   Allocate host page-lock memory.
 *
 * @param[in]   size     size in bytes to copy.
 *
 * @return      dest_h   pointer to host memory.
 */
extern void *hl_malloc_host(size_t size);

/**
 * @brief   Free host page-lock memory.
 *
 * @param[in]   dest_h  pointer to host memory.
 *
 */
extern void hl_free_mem_host(void *dest_h);

/**
 * @brief   Copy data.
 *
 * @param[in]   dst     dst memory address(host or device).
 * @param[in]   src     src memory address(host or device).
 * @param[in]   size    size in bytes to copy.
 *
 */
extern void hl_memcpy(void *dst, void *src, size_t size);

/**
 * @brief   Set device memory to a value.
 *
 * @param[in]   dest_d  pointer to device memory.
 * @param[in]   value   value to set for each byte of specified memory.
 * @param[in]   size    size in bytes to set.
 *
 */
extern void hl_memset_device(void *dest_d, int value, size_t size);

/**
 * @brief   Copy host memory to device memory.
 *
 * @param[in]   dest_d  dst memory address.
 * @param[in]   src_h   src memory address.
 * @param[in]   size    size in bytes to copy.
 *
 */
extern void hl_memcpy_host2device(void *dest_d, void *src_h, size_t size);

/**
 * @brief   Copy device memory to host memory.
 *
 * @param[in]   dest_h  dst memory address.
 * @param[in]   src_d   src memory address.
 * @param[in]   size    size in bytes to copy.
 *
 */
extern void hl_memcpy_device2host(void *dest_h, void *src_d, size_t size);

/**
 * @brief   Copy device memory to device memory.
 *
 * @param[in]   dest_d  dst memory address.
 * @param[in]   src_d   src memory address.
 * @param[in]   size    size in bytes to copy.
 *
 */
extern void hl_memcpy_device2device(void *dest_d, void *src_d, size_t size);

/**
 * @brief   Generate uniformly distributed floats (0, 1.0].
 *
 * @param[in]   dest_d  pointer to device memory to store results.
 * @param[in]   num     number of floats to generate.
 *
 */
extern void hl_rand(real *dest_d, size_t num);

/**
 * @brief   Set the seed value of the random number generator.
 *
 * @param[in]   seed    seed value.
 */
extern void hl_srand(unsigned int seed);

/**
 * @brief   Copy data.
 *
 * @param[in]   dst     dst memory address(host or device).
 * @param[in]   src     src memory address(host or device).
 * @param[in]   size    size in bytes to copy.
 * @param[in]   stream  stream id.
 */
extern void hl_memcpy_async(void *dst,
                            void *src,
                            size_t size,
                            hl_stream_t stream);

/**
 * @brief   Waits for stream tasks to complete.
 *
 * @param[in]   stream  stream id.
 */
extern void hl_stream_synchronize(hl_stream_t stream);

/**
 * @brief   Creates an event object.
 *
 * @param[out]   event  New event.
 */
extern void hl_create_event(hl_event_t *event);

/**
 * @brief   Destroys an event object.
 *
 * @param[in]   event   Event to destroy.
 */
extern void hl_destroy_event(hl_event_t event);

/**
 * @brief   Computes the elapsed time between events.
 *
 * @param[in]   start  Starting event.
 * @param[in]   end    Ending event.
 *
 * @return      time   Time between start and end in ms.
 */
extern float hl_event_elapsed_time(hl_event_t start, hl_event_t end);

/**
 * @brief   Records an event.
 *
 * @param[in]   stream   Stream in which to insert event.
 * @param[in]   event    Event waiting to be recorded as completed.
 *
 */
extern void hl_stream_record_event(hl_stream_t stream, hl_event_t event);

/**
 * @brief   Make a compute stream wait on an event.
 *
 * @param[in]   stream   Stream in which to insert event.
 * @param[in]   event    Event to wait on.
 *
 */
extern void hl_stream_wait_event(hl_stream_t stream, hl_event_t event);

/**
 * @brief   Wait for an event to complete.
 *
 * @param[in]   event       event to wait for.
 *
 */
extern void hl_event_synchronize(hl_event_t event);

/**
 * @brief   Sets block flags to be used for device executions.
 *
 * @note    This interface needs to be called before hl_start.
 */
extern void hl_set_device_flags_block();

/**
 * @brief   Returns the last error string from a cuda runtime call.
 */
extern const char *hl_get_device_error_string();

/**
 * @brief     Returns the last error string from a cuda runtime call.
 *
 * @param[in] err  error number.
 *
 * @see       hl_get_device_last_error()
 */
extern const char *hl_get_device_error_string(size_t err);

/**
 * @brief   Returns the last error number.
 *
 * @return  error number.
 *
 * @see     hl_get_device_error_string()
 */
extern int hl_get_device_last_error();

/**
 * @brief   check cuda event is ready
 *
 * @param[in]  event        cuda event to query.
 *
 * @return     true    cuda event is ready.
 *             false   cuda event is not ready.
 */
extern bool hl_cuda_event_is_ready(hl_event_t event);

/**
 * @brief   hppl device synchronization.
 */
extern void hl_device_synchronize();

/**
 * @brief   gpu profiler start
 */
extern void hl_profiler_start();

/**
 * @brief   gpu profiler stop
 */
extern void hl_profiler_end();

#endif  // HL_CUDA_H_
