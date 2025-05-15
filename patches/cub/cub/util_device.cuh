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

CUB_NAMESPACE_BEGIN


/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document


/**
 * \brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t& temp_storage_bytes,                 ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return CubDebug(cudaErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *) ((size_t(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return cudaSuccess;
}


/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
CUB_RUNTIME_FUNCTION inline int CurrentDevice()
{
    int device = -1;
    if (CubDebug(cudaGetDevice(&device))) return -1;
    return device;
}

/**
 * \brief RAII helper which saves the current device and switches to the
 *        specified device on construction and switches to the saved device on
 *        destruction.
 */
struct SwitchDevice
{
private:
    int const old_device;
    bool const needs_reset;
public:
    __host__ inline SwitchDevice(int new_device)
      : old_device(CurrentDevice()), needs_reset(old_device != new_device)
    {
        if (needs_reset)
            CubDebug(cudaSetDevice(new_device));
    }

    __host__ inline ~SwitchDevice()
    {
        if (needs_reset)
            CubDebug(cudaSetDevice(old_device));
    }
};

/**
 * \brief Returns the number of CUDA devices available or -1 if an error
 *        occurred.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCountUncached()
{
    int count = -1;
    if (CubDebug(cudaGetDeviceCount(&count)))
        // CUDA makes no guarantees about the state of the output parameter if
        // `cudaGetDeviceCount` fails; in practice, they don't, but out of
        // paranoia we'll reset `count` to `-1`.
        count = -1;
    return count;
}

/**
 * \brief Cache for an arbitrary value produced by a nullary function.
 */
template <typename T, T(*Function)()>
struct ValueCache
{
    T const value;

    /**
     * \brief Call the nullary function to produce the value and construct the
     *        cache.
     */
    __host__ inline ValueCache() : value(Function()) {}
};

// Host code, only safely usable in C++11 or newer, where thread-safe
// initialization of static locals is guaranteed.  This is a separate function
// to avoid defining a local static in a host/device function.
__host__ inline int DeviceCountCachedValue()
{
    static ValueCache<int, DeviceCountUncached> cache;
    return cache.value;
}

/**
 * \brief Returns the number of CUDA devices available.
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCount()
{
    int result = -1;

    NV_IF_TARGET(NV_IS_HOST,
                 (result = DeviceCountCachedValue();),
                 (result = DeviceCountUncached();));

    return result;
}

/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
    struct DevicePayload
    {
        int         attribute;
        cudaError_t error;
    };

    // Each entry starts in the `DeviceEntryEmpty` state, then proceeds to the
    // `DeviceEntryInitializing` state, and then proceeds to the
    // `DeviceEntryReady` state. These are the only state transitions allowed;
    // e.g. a linear sequence of transitions.
    enum DeviceEntryStatus
    {
        DeviceEntryEmpty = 0,
        DeviceEntryInitializing,
        DeviceEntryReady
    };

    struct DeviceEntry
    {
        std::atomic<DeviceEntryStatus> flag;
        DevicePayload                  payload;
    };

private:
    std::array<DeviceEntry, CUB_MAX_DEVICES> entries_;

public:
    /**
     * \brief Construct the cache.
     */
    __host__ inline PerDeviceAttributeCache() : entries_()
    {
        assert(DeviceCount() <= CUB_MAX_DEVICES);
    }

    /**
     * \brief Retrieves the payload of the cached function \p f for \p device.
     *
     * \note You must pass a morally equivalent function in to every call or
     *       this function has undefined behavior.
     */
    template <typename Invocable>
    __host__ DevicePayload operator()(Invocable&& f, int device)
    {
        if (device >= DeviceCount())
            return DevicePayload{0, cudaErrorInvalidDevice};

        auto& entry   = entries_[device];
        auto& flag    = entry.flag;
        auto& payload = entry.payload;

        DeviceEntryStatus old_status = DeviceEntryEmpty;

        // First, check for the common case of the entry being ready.
        if (flag.load(std::memory_order_acquire) != DeviceEntryReady)
        {
            // Assume the entry is empty and attempt to lock it so we can fill
            // it by trying to set the state from `DeviceEntryReady` to
            // `DeviceEntryInitializing`.
            if (flag.compare_exchange_strong(old_status, DeviceEntryInitializing,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // We successfully set the state to `DeviceEntryInitializing`;
                // we have the lock and it's our job to initialize this entry
                // and then release it.

                // We don't use `CubDebug` here because we let the user code
                // decide whether or not errors are hard errors.
                payload.error = ::cuda::std::forward<Invocable>(f)(payload.attribute);
                if (payload.error)
                    // Clear the global CUDA error state which may have been
                    // set by the last call. Otherwise, errors may "leak" to
                    // unrelated kernel launches.
                    cudaGetLastError();

                // Release the lock by setting the state to `DeviceEntryReady`.
                flag.store(DeviceEntryReady, std::memory_order_release);
            }

            // If the `compare_exchange_weak` failed, then `old_status` has
            // been updated with the value of `flag` that it observed.

            else if (old_status == DeviceEntryInitializing)
            {
                // Another execution agent is initializing this entry; we need
                // to wait for them to finish; we'll know they're done when we
                // observe the entry status as `DeviceEntryReady`.
                do { old_status = flag.load(std::memory_order_acquire); }
                while (old_status != DeviceEntryReady);
                // FIXME: Use `atomic::wait` instead when we have access to
                // host-side C++20 atomics. We could use libcu++, but it only
                // supports atomics for SM60 and up, even if you're only using
                // them in host code.
            }
        }

        // We now know that the state of our entry is `DeviceEntryReady`, so
        // just return the entry's payload.
        return entry.payload;
    }
};

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersionUncached(int& ptx_version)
{
    // Instantiate `EmptyKernel<void>` in both host and device code to ensure
    // it can be called.
    typedef void (*EmptyKernelPtr)();
    EmptyKernelPtr empty_kernel = EmptyKernel<void>;

    // This is necessary for unused variable warnings in host compilers. The
    // usual syntax of (void)empty_kernel; was not sufficient on MSVC2015.
    (void)reinterpret_cast<void*>(empty_kernel);

    // Define a temporary macro that expands to the current target ptx version
    // in device code.
    // <nv/target> may provide an abstraction for this eventually. For now,
    // we have to keep this usage of __CUDA_ARCH__.
#if defined(_NVHPC_CUDA)
#define CUB_TEMP_GET_PTX __builtin_current_device_sm()
#else
#define CUB_TEMP_GET_PTX __CUDA_ARCH__
#endif

    cudaError_t result = cudaSuccess;
    NV_IF_TARGET(
      NV_IS_HOST,
      (
        cudaFuncAttributes empty_kernel_attrs;

        result = cudaFuncGetAttributes(&empty_kernel_attrs,
                                       reinterpret_cast<void*>(empty_kernel));
        CubDebug(result);

        ptx_version = empty_kernel_attrs.ptxVersion * 10;
      ),
      // NV_IS_DEVICE
      (
        // This is necessary to ensure instantiation of EmptyKernel in device
        // code. The `reinterpret_cast` is necessary to suppress a
        // set-but-unused warnings. This is a meme now:
        // https://twitter.com/blelbach/status/1222391615576100864
        (void)reinterpret_cast<EmptyKernelPtr>(empty_kernel);

        ptx_version = CUB_TEMP_GET_PTX;
      ));

#undef CUB_TEMP_GET_PTX

    return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
__host__ inline cudaError_t PtxVersionUncached(int& ptx_version, int device)
{
    SwitchDevice sd(device);
    (void)sd;
    return PtxVersionUncached(ptx_version);
}

template <typename Tag>
__host__ inline PerDeviceAttributeCache& GetPerDeviceAttributeCache()
{
    // C++11 guarantees that initialization of static locals is thread safe.
    static PerDeviceAttributeCache cache;
    return cache;
}

struct PtxVersionCacheTag {};
struct SmVersionCacheTag {};

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
__host__ inline cudaError_t PtxVersion(int& ptx_version, int device)
{
    auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
      // If this call fails, then we get the error code back in the payload,
      // which we check with `CubDebug` below.
      [=] (int& pv) { return PtxVersionUncached(pv, device); },
      device);

    if (!CubDebug(payload.error))
        ptx_version = payload.attribute;

    return payload.error;
}

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersion(int &ptx_version)
{
  cudaError_t result = cudaErrorUnknown;
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      auto const device  = CurrentDevice();
      auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
        // If this call fails, then we get the error code back in the payload,
        // which we check with `CubDebug` below.
        [=](int &pv) { return PtxVersionUncached(pv, device); },
        device);

      if (!CubDebug(payload.error))
      {
        ptx_version = payload.attribute;
      }

      result = payload.error;
    ),
    ( // NV_IS_DEVICE:
      result = PtxVersionUncached(ptx_version);
    ));

  return result;
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersionUncached(int& sm_version, int device = CurrentDevice())
{
    cudaError_t error = cudaSuccess;
    do
    {
        int major = 0, minor = 0;
        if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device))) break;
        if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device))) break;
        sm_version = major * 100 + minor * 10;
    }
    while (0);

    return error;
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersion(int &sm_version,
                                                  int device = CurrentDevice())
{
  cudaError_t result = cudaErrorUnknown;

  NV_IF_TARGET(
    NV_IS_HOST,
    (
      auto const payload = GetPerDeviceAttributeCache<SmVersionCacheTag>()(
      // If this call fails, then we get the error code back in
      // the payload, which we check with `CubDebug` below.
      [=](int &pv) { return SmVersionUncached(pv, device); },
      device);

      if (!CubDebug(payload.error))
      {
        sm_version = payload.attribute;
      };

      result = payload.error;
    ),
    ( // NV_IS_DEVICE
      result = SmVersionUncached(sm_version, device);
    ));

  return result;
}

/**
 * Synchronize the specified \p stream.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SyncStream(cudaStream_t stream)
{
  cudaError_t result = cudaErrorNotSupported;

  NV_IF_TARGET(NV_IS_HOST,
               (result = CubDebug(cudaStreamSynchronize(stream));),
               ((void)stream;
                result = CubDebug(cub::detail::device_synchronize());));

  return result;
}

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


namespace detail
{

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

/** \brief Gets whether the current device supports unified addressing */
CUB_RUNTIME_FUNCTION inline cudaError_t HasUVA(bool& has_uva)
{
    has_uva = false;
    cudaError_t error = cudaSuccess;
    int device = -1;
    if (CubDebug(error = cudaGetDevice(&device)) != cudaSuccess) return error;
    int uva = 0;
    if (CubDebug(error = cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device))
        != cudaSuccess)
    {
        return error;
    }
    has_uva = uva == 1;
    return error;
}

} // namespace detail

/**
 * \brief Computes maximum SM occupancy in thread blocks for executing the given kernel function pointer \p kernel_ptr on the current device with \p block_threads per thread block.
 *
 * \par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * \endcode
 *
 */
template <typename KernelPtr>
CUB_RUNTIME_FUNCTION inline
cudaError_t MaxSmOccupancy(
    int&                max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads,              ///< [in] Number of threads per thread block
    int                 dynamic_smem_bytes = 0)	    ///< [in] Dynamically allocated shared memory in bytes. Default is 0.
{
    return CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        kernel_ptr,
        block_threads,
        dynamic_smem_bytes));
}


/******************************************************************************
 * Policy management
 ******************************************************************************/

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_size;
    int sm_occupancy;

    CUB_RUNTIME_FUNCTION __forceinline__
    KernelConfig() : block_threads(0), items_per_thread(0), tile_size(0), sm_occupancy(0) {}

    template <typename AgentPolicyT, typename KernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Init(KernelPtrT kernel_ptr)
    {
        block_threads        = AgentPolicyT::BLOCK_THREADS;
        items_per_thread     = AgentPolicyT::ITEMS_PER_THREAD;
        tile_size            = block_threads * items_per_thread;
        cudaError_t retval   = MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
        return retval;
    }
};



/// Helper for dispatching into a policy chain
template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
  /// The policy for the active compiler pass
  using ActivePolicy =
    cub::detail::conditional_t<(CUB_PTX_ARCH < PTX_VERSION),
                               typename PrevPolicyT::ActivePolicy,
                               PolicyT>;

  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION __forceinline__
  static cudaError_t Invoke(int ptx_version, FunctorT& op)
  {
      if (ptx_version < PTX_VERSION) {
          return PrevPolicyT::Invoke(ptx_version, op);
      }
      return op.template Invoke<PolicyT>();
  }
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PTX_VERSION, typename PolicyT>
struct ChainedPolicy<PTX_VERSION, PolicyT, PolicyT>
{
    /// The policy for the active compiler pass
    typedef PolicyT ActivePolicy;

    /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
    template <typename FunctorT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Invoke(int /*ptx_version*/, FunctorT& op) {
        return op.template Invoke<PolicyT>();
    }
};




/** @} */       // end group UtilMgmt

CUB_NAMESPACE_END