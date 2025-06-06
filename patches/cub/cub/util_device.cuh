/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include <cuda/std/utility>

#include <cub/detail/device_synchronize.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_cpp_dialect.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <nv/target>

#include <atomic>
#include <array>
#include <cassert>


namespace cub
{

/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document


/**
 * Same as SyncStream, but intended for use with the debug_synchronous flags
 * in device algorithms. This should not be used if synchronization is required
 * for correctness.
 *
 * If `debug_synchronous` is false, this function will immediately return
 * cudaSuccess. If true, one of the following will occur:
 *
 * If synchronization is supported by the current compilation target and
 * settings, the sync is performed and the sync result is returned.
 *
 * If syncs are not supported then no sync is performed, but a message is logged
 * via _CubLog and cudaSuccess is returned.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t DebugSyncStream(cudaStream_t stream)
{
#ifndef CUB_DETAIL_DEBUG_ENABLE_SYNC

  (void)stream;
  return cudaSuccess;

#else // CUB_DETAIL_DEBUG_ENABLE_SYNC:

#define CUB_TMP_SYNC_AVAILABLE                                                 \
  _CubLog("%s\n", "Synchronizing...");                                         \
  return SyncStream(stream)

#define CUB_TMP_DEVICE_SYNC_UNAVAILABLE                                        \
  (void)stream;                                                                \
  _CubLog("WARNING: Skipping CUB `debug_synchronous` synchronization (%s).\n", \
          "device-side sync requires <sm_90, RDC, and CDPv1");                 \
  return cudaSuccess

#ifdef CUB_DETAIL_CDPv1

  // Can sync everywhere but SM_90+
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (CUB_TMP_DEVICE_SYNC_UNAVAILABLE;),
               (CUB_TMP_SYNC_AVAILABLE;));

#else // CDPv2 or no CDP:

  // Can only sync on host
  NV_IF_TARGET(NV_IS_HOST,
               (CUB_TMP_SYNC_AVAILABLE;),
               (CUB_TMP_DEVICE_SYNC_UNAVAILABLE;));

#endif // CDP version

#undef CUB_TMP_DEVICE_SYNC_UNAVAILABLE
#undef CUB_TMP_SYNC_AVAILABLE

#endif // CUB_DETAIL_DEBUG_ENABLE_SYNC
}

#endif/** @} */       // end group UtilMgmt

} /* end namespace cub */
