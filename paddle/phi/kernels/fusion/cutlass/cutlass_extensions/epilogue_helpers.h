/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can
 * signal to template code the type of epilogue we want to run, and let the
 * underlying code specify the details such as element types, accumulator type
 * and elements per vector access.
 *
 */

#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "epilogue/thread/ft_fused_activations.h"

namespace phi {
struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpNoBias {};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator,
          typename Op>
struct Epilogue {};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBiasSilu> {
  using Op = cutlass::epilogue::thread::LinearCombinationSilu<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBiasReLU> {
  using Op = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBiasFtGelu> {
  using Op = cutlass::epilogue::thread::LinearCombinationGeneric<
      cutlass::epilogue::thread::GELU_taylor,
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling,
      cutlass::FloatRoundStyle::round_to_nearest,
      true>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType,
          int ElementsPerVectorAccess,
          typename ElementAccumulator>
struct Epilogue<ElementType,
                ElementsPerVectorAccess,
                ElementAccumulator,
                EpilogueOpNoBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<
      ElementType,
      ElementsPerVectorAccess,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::Default>;
};
}  // namespace phi
