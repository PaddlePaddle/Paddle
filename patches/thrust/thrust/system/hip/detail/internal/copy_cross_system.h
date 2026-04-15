#pragma once

// Full shadow of rocThrust's copy_cross_system.h for ROCm 7 under HIP-clang.
//
// Problem:
// rocThrust uses `NV_IF_TARGET(...)` inside functions that are annotated as both
// host and device (e.g. `THRUST_HOST /*WORKAROUND*/ THRUST_DEVICE`).
// Under HIP-clang, the "host" branch can still be parsed/instantiated for device
// compilation, causing errors like:
//   reference to __host__ function 'trivial_device_copy' in __host__ __device__ function
//
// Fix:
// Replace that `NV_IF_TARGET` guard with a hard preprocessor split based on
// `__HIP_DEVICE_COMPILE__`, which is reliable for HIP compilation modes.

// NOTE: This file intentionally mirrors ROCm 7 rocThrust (rocm-7.0.0) layout.

#include <thrust/detail/config.h>

// XXX
// this file must not be included on its own, ever,
// but must be part of include in thrust/system/hip/detail/copy.h

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/temporary_array.h>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace __copy
{
template <class T, class Size>
THRUST_HIP_HOST_FUNCTION void trivial_device_copy(
  thrust::cpp::execution_policy &,
  thrust::hip_rocprim::execution_policy & device_s,
  T* dst,
  T const* src,
  Size count)
{
  hipError_t status;
  status = hip_rocprim::trivial_copy_to_device(dst, src, count, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy::trivial_device_copy H->D: failed");
}

template <class T, class Size>
THRUST_HIP_HOST_FUNCTION void trivial_device_copy(
  thrust::hip_rocprim::execution_policy & device_s,
  thrust::cpp::execution_policy &,
  T* dst,
  T const* src,
  Size count)
{
  hipError_t status;
  status = hip_rocprim::trivial_copy_from_device(dst, src, count, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "trivial_device_copy D->H failed");
}

template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HOST /* WORKAROUND */ THRUST_DEVICE cross_system_copy_n(
  thrust::execution_policy & sys1,
  thrust::execution_policy & sys2,
  InputIt begin,
  Size n,
  OutputIt result,
  thrust::detail::true_type) // trivial copy
{
#if defined(__HIP_DEVICE_COMPILE__)
  THRUST_UNUSED_VAR(sys1);
  THRUST_UNUSED_VAR(sys2);
  THRUST_UNUSED_VAR(n);
  THRUST_UNUSED_VAR(begin);
  return result;
#else
  using InputTy = typename iterator_traits<InputIt>::value_type;
  if (n > 0)
  {
    trivial_device_copy(
      derived_cast(sys1),
      derived_cast(sys2),
      reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
      reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*begin)),
      static_cast<Size>(n));
  }
  return result + n;
#endif
}

// non-trivial H->D copy
template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HIP_RUNTIME_FUNCTION cross_system_copy_n_hd_nt(
  thrust::cpp::execution_policy & host_s,
  thrust::hip_rocprim::execution_policy & device_s,
  InputIt first,
  Size num_items,
  OutputIt result)
{
  using InputTy = typename thrust::iterator_value<InputIt>::type;

  InputIt last = first;
  thrust::advance(last, num_items);
  thrust::detail::temporary_array<InputTy> temp(host_s, num_items);

  for (Size idx = 0; idx != num_items; idx++)
  {
    ::new (static_cast<void*>(temp.data().get() + idx)) InputTy(*first);
    ++first;
  }

  thrust::detail::temporary_array<InputTy> d_in_ptr(device_s, num_items);

  hipError_t status = hip_rocprim::trivial_copy_to_device(
    d_in_ptr.data().get(), temp.data().get(), num_items, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy:: H->D: failed");

  OutputIt ret = hip_rocprim::copy_n(device_s, d_in_ptr.data(), num_items, result);
  return ret;
}

template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HIP_FUNCTION cross_system_copy_n(
  thrust::cpp::execution_policy & host_s,
  thrust::hip_rocprim::execution_policy & device_s,
  InputIt first,
  Size num_items,
  OutputIt result,
  thrust::detail::false_type)
{
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(thrust::cpp::execution_policy & host_s,
        thrust::hip_rocprim::execution_policy & device_s,
        InputIt first,
        Size num_items,
        OutputIt result)
    {
      return cross_system_copy_n_hd_nt(host_s, device_s, first, num_items, result);
    }

    THRUST_DEVICE static OutputIt
    seq(thrust::cpp::execution_policy & host_s,
        thrust::hip_rocprim::execution_policy & device_s,
        InputIt first,
        Size num_items,
        OutputIt result)
    {
      THRUST_UNUSED_VAR(host_s);
      THRUST_UNUSED_VAR(device_s);
      THRUST_UNUSED_VAR(first);
      THRUST_UNUSED_VAR(num_items);
      return result;
    }
  };

#if __THRUST_HAS_HIPRT__
  return workaround::par(host_s, device_s, first, num_items, result);
#else
  return workaround::seq(host_s, device_s, first, num_items, result);
#endif
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HIP_RUNTIME_FUNCTION cross_system_copy_n_dh_nt(
  thrust::hip_rocprim::execution_policy & device_s,
  thrust::cpp::execution_policy & host_s,
  InputIt first,
  Size num_items,
  OutputIt result)
{
  using InputTy = typename thrust::iterator_value<InputIt>::type;

  thrust::detail::temporary_array<InputTy> d_in_ptr(device_s, num_items);
  hip_rocprim::uninitialized_copy_n(device_s, first, num_items, d_in_ptr.data());

  thrust::detail::temporary_array<InputTy> temp(host_s, num_items);

  hipError_t status = hip_rocprim::trivial_copy_from_device(
    temp.data().get(), d_in_ptr.data().get(), num_items, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy:: D->H: failed");

  OutputIt ret = thrust::copy_n(host_s, temp.data(), num_items, result);
  return ret;
}

template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HIP_FUNCTION cross_system_copy_n(
  thrust::hip_rocprim::execution_policy & device_s,
  thrust::cpp::execution_policy & host_s,
  InputIt first,
  Size num_items,
  OutputIt result,
  thrust::detail::false_type)
{
  struct workaround
  {
    THRUST_HOST static void
    par(thrust::hip_rocprim::execution_policy & device_s,
        thrust::cpp::execution_policy & host_s,
        InputIt first,
        Size num_items,
        OutputIt& result)
    {
      result = cross_system_copy_n_dh_nt(device_s, host_s, first, num_items, result);
    }

    THRUST_DEVICE static void
    seq(thrust::hip_rocprim::execution_policy & device_s,
        thrust::cpp::execution_policy & host_s,
        InputIt first,
        Size num_items,
        OutputIt& result)
    {
      THRUST_UNUSED_VAR(device_s);
      THRUST_UNUSED_VAR(host_s);
      THRUST_UNUSED_VAR(first);
      THRUST_UNUSED_VAR(num_items);
      THRUST_UNUSED_VAR(result);
    }
  };

# if __THRUST_HAS_HIPRT__
  workaround::par(device_s, host_s, first, num_items, result);
  return result;
# else
  workaround::seq(device_s, host_s, first, num_items, result);
  return result;
# endif
}
#endif

template <class InputIt, class Size, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
cross_system_copy_n(cross_system systems, InputIt begin, Size n, OutputIt result)
{
  return cross_system_copy_n(
    derived_cast(systems.sys1),
    derived_cast(systems.sys2),
    begin,
    n,
    result,
    typename is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::type());
}

template <class InputIterator, class OutputIterator>
OutputIterator THRUST_HIP_FUNCTION
cross_system_copy(cross_system systems, InputIterator begin, InputIterator end, OutputIterator result)
{
  return cross_system_copy_n(systems, begin, static_cast<decltype(thrust::distance(begin, end))>(thrust::distance(begin, end)), result);
}
} // namespace __copy
} // namespace hip_rocprim
THRUST_NAMESPACE_END

