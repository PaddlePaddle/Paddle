/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <type_traits>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/enforce.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "thrust/device_ptr.h"
#endif

namespace phi {

// Transform applies a unary or a binary functor on each element in a
// range defined by a pair of iterators.
//
// - The specialization for CPU calls std::transform.
// - The specialization for CUDA calls thrust::transform.
//
// NOTE: We need to define InputIter and OutputIter defined as
//       different types, because the InputIter points op's inputs and
//       OutputIter points to op's outputs.
//
// NOTE: We don't assume that InputIter to be const InputType* and
//       OutputIter to be OutputType*, because we might use a iterator
//       class, paddle::fluid::operators::RowwiseTransformIterator.
template <typename Context>
struct Transform {
  // The unary version.
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const Context& context,
                  InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op);

  // The binary version.
  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(const Context& context,
                  InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op);
};

// NOTE: After the phi kernel is migrated, it needs to be deleted.

template <>
struct Transform<phi::CPUContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const phi::CPUContext& context UNUSED,
                  InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(const phi::CPUContext& context UNUSED,
                  InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)

// PointerToThrustDevicePtr has two specializations, one casts a (CUDA
// device) pointer into thrust::device_ptr, the other keeps rest types
// un-casted.
template <typename T, bool is_ptr>
struct PointerToThrustDevicePtr;

template <typename T>
struct PointerToThrustDevicePtr<T, true> {
  using ELEM = typename std::remove_pointer<T>::type;
  using RTYPE = thrust::device_ptr<ELEM>;

  inline thrust::device_ptr<ELEM> operator()(ELEM* ele) const {
    return thrust::device_pointer_cast(ele);
  }
};

template <typename T>
struct PointerToThrustDevicePtr<T, false> {
  using RTYPE = T;
  inline RTYPE operator()(RTYPE it) const { return it; }
};

// CastToCUDATransformIterator casts a pointer to thrust::device_ptr
// so it could be used as the iterator of thrust::transform.  It
// doesn't cast other types.
//
// We need CastToCUDATransformIterator because it is often that we
// want to use device memory pointers as transform iterators, e.g., to
// transform a block of float32 to float16.  In this case, we want
// CastToCUDATransformIterator to cast float16/32 pointers to
// thrust::device_ptr, otherwise they cannot work as the iterator
// required by thrust::transform.  At the same time, we don't want to
// cast thrust::device_ptr to thrust::device_ptr repeatedly.
template <typename T>
auto CastToCUDATransformIterator(T t) ->
    typename PointerToThrustDevicePtr<T, std::is_pointer<T>::value>::RTYPE {
  PointerToThrustDevicePtr<T, std::is_pointer<T>::value> cast;
  return cast(t);
}

template <>
struct Transform<phi::GPUContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const phi::GPUContext& context,
                  InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::GPU,
                      true,
                      phi::errors::PreconditionNotMet(
                          "The CUDA Transform must be used in GPU place."));
#ifdef __HIPCC__
    thrust::transform(thrust::hip::par.on(context.stream()),
                      CastToCUDATransformIterator(first),
                      CastToCUDATransformIterator(last),
                      CastToCUDATransformIterator(result),
                      op);
#else
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      CastToCUDATransformIterator(first),
                      CastToCUDATransformIterator(last),
                      CastToCUDATransformIterator(result),
                      op);
#endif
  }

  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(const phi::GPUContext& context,
                  InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::GPU,
                      true,
                      phi::errors::PreconditionNotMet(
                          "The CUDA Transform must be used in GPU place."));
#ifdef __HIPCC__
    thrust::transform(thrust::hip::par.on(context.stream()),
                      CastToCUDATransformIterator(first1),
                      CastToCUDATransformIterator(last1),
                      CastToCUDATransformIterator(first2),
                      CastToCUDATransformIterator(result),
                      op);
#else
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      CastToCUDATransformIterator(first1),
                      CastToCUDATransformIterator(last1),
                      CastToCUDATransformIterator(first2),
                      CastToCUDATransformIterator(result),
                      op);
#endif
  }
};
#endif

}  // namespace phi
