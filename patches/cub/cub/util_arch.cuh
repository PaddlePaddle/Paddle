/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * Static architectural properties by SM version.
 */

#pragma once

#include <cub/util_cpp_dialect.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_macro.cuh>


namespace  cub
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

// \deprecated [Since 2.1.0] 
#define CUB_USE_COOPERATIVE_GROUPS

/// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
/// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
#ifndef CUB_PTX_ARCH
    #if defined(_NVHPC_CUDA)
        // __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
        // when compiling both host code and device code. Currently, only one
        // PTX version can be targeted.
        #define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
    #elif !defined(__CUDA_ARCH__)
        #define CUB_PTX_ARCH 0
    #else
        #define CUB_PTX_ARCH __CUDA_ARCH__
    #endif
#endif

// These definitions were intended for internal use only and are now obsolete.
// If you relied on them, consider porting your code to use the functionality
// in libcu++'s <nv/target> header.
// For a temporary workaround, define CUB_PROVIDE_LEGACY_ARCH_MACROS to make
// them available again. These should be considered deprecated and will be
// fully removed in a future version.
#ifdef CUB_PROVIDE_LEGACY_ARCH_MACROS
    #ifndef CUB_IS_DEVICE_CODE
        #if defined(_NVHPC_CUDA)
            #define CUB_IS_DEVICE_CODE __builtin_is_device_code()
            #define CUB_IS_HOST_CODE (!__builtin_is_device_code())
            #define CUB_INCLUDE_DEVICE_CODE 1
            #define CUB_INCLUDE_HOST_CODE 1
        #elif CUB_PTX_ARCH > 0
            #define CUB_IS_DEVICE_CODE 1
            #define CUB_IS_HOST_CODE 0
            #define CUB_INCLUDE_DEVICE_CODE 1
            #define CUB_INCLUDE_HOST_CODE 0
        #else
            #define CUB_IS_DEVICE_CODE 0
            #define CUB_IS_HOST_CODE 1
            #define CUB_INCLUDE_DEVICE_CODE 0
            #define CUB_INCLUDE_HOST_CODE 1
        #endif
    #endif
#endif // CUB_PROVIDE_LEGACY_ARCH_MACROS

/// Maximum number of devices supported.
#ifndef CUB_MAX_DEVICES
    #define CUB_MAX_DEVICES (128)
#endif

static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");


// Whether the current compilation pass supports calling cudaDeviceSynchronize
// from device code.
#ifndef CUB_TARGET_ALLOWS_DEVICE_SYNC
    #if !defined(__CUDA_ARCH__) || (defined(CUB_RUNTIME_ENABLED) && __CUDA_ARCH__ < 900)
        #define CUB_TARGET_ALLOWS_DEVICE_SYNC 1
    #else
        #define CUB_TARGET_ALLOWS_DEVICE_SYNC 0
    #endif
#endif

/// Number of threads per warp
#ifndef CUB_LOG_WARP_THREADS
    #define CUB_LOG_WARP_THREADS(unused) (5)
    #define CUB_WARP_THREADS(unused) (1 << CUB_LOG_WARP_THREADS(0))

    #define CUB_PTX_WARP_THREADS        CUB_WARP_THREADS(0)
    #define CUB_PTX_LOG_WARP_THREADS    CUB_LOG_WARP_THREADS(0)
#endif


/// Number of smem banks
#ifndef CUB_LOG_SMEM_BANKS
    #define CUB_LOG_SMEM_BANKS(unused) (5)
    #define CUB_SMEM_BANKS(unused) (1 << CUB_LOG_SMEM_BANKS(0))

    #define CUB_PTX_LOG_SMEM_BANKS      CUB_LOG_SMEM_BANKS(0)
    #define CUB_PTX_SMEM_BANKS          CUB_SMEM_BANKS
#endif


/// Oversubscription factor
#ifndef CUB_SUBSCRIPTION_FACTOR
    #define CUB_SUBSCRIPTION_FACTOR(unused) (5)
    #define CUB_PTX_SUBSCRIPTION_FACTOR CUB_SUBSCRIPTION_FACTOR(0)
#endif


/// Prefer padding overhead vs X-way conflicts greater than this threshold
#ifndef CUB_PREFER_CONFLICT_OVER_PADDING
    #define CUB_PREFER_CONFLICT_OVER_PADDING(unused) (1)
    #define CUB_PTX_PREFER_CONFLICT_OVER_PADDING CUB_PREFER_CONFLICT_OVER_PADDING(0)
#endif

#endif  // Do not document

} /* end namespace cub */
