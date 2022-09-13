/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*
  \file
  \brief Defines a data structure in which a set of functionally equivalent library::Operation
        instances may be queried.
*/

#pragma once
#include <fstream>
#include <iosfwd>
#include <unordered_map>
#include <algorithm>

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/util.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for Gemm Functional Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tuple uniquely identifying Gemm functional behavior
struct GemmFunctionalKey {

  Provider provider;
  GemmKind gemm_kind;
  NumericTypeID element_compute;
  NumericTypeID element_scalar;
  NumericTypeID element_A;
  LayoutTypeID layout_A;
  ComplexTransform transform_A;
  NumericTypeID element_B;
  LayoutTypeID layout_B;
  ComplexTransform transform_B;
  NumericTypeID element_C;

  //
  // Methods
  //

  inline
  GemmFunctionalKey(
    Provider provider,
    GemmKind gemm_kind = GemmKind::kGemm,
    NumericTypeID element_compute = NumericTypeID::kF32,
    NumericTypeID element_scalar = NumericTypeID::kF32,
    NumericTypeID element_A = NumericTypeID::kF16,
    LayoutTypeID layout_A = LayoutTypeID::kColumnMajor,
    ComplexTransform transform_A = ComplexTransform::kNone,
    NumericTypeID element_B = NumericTypeID::kF16,
    LayoutTypeID layout_B = LayoutTypeID::kColumnMajor,
    ComplexTransform transform_B = ComplexTransform::kNone,
    NumericTypeID element_C = NumericTypeID::kF16
  ):
    provider(provider),
    gemm_kind(gemm_kind),
    element_compute(element_compute),
    element_scalar(element_scalar),
    element_A(element_A),
    layout_A(layout_A),
    transform_A(transform_A),
    element_B(element_B),
    layout_B(layout_B),
    transform_B(transform_B),
    element_C(element_C)
  { }

  inline
  bool operator==(GemmFunctionalKey const &rhs) const {
    return 
      (provider == rhs.provider) &&
      (gemm_kind == rhs.gemm_kind) &&
      (element_compute == rhs.element_compute) &&
      (element_scalar == rhs.element_scalar) &&
      (element_A == rhs.element_A) &&
      (layout_A == rhs.layout_A) &&
      (transform_A == rhs.transform_A) &&
      (element_B == rhs.element_B) &&
      (layout_B == rhs.layout_B) &&
      (transform_B == rhs.transform_B) &&
      (element_C == rhs.element_C);
  }

  inline
  bool operator!=(GemmFunctionalKey const &rhs) const {
    return !(*this == rhs);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
inline
std::ostream & operator<<(std::ostream &out, cutlass::library::GemmFunctionalKey const &k) {

  out << "{\n"
    << "         provider: " << to_string(k.provider) << "\n"
    << "        gemm_kind: " << to_string(k.gemm_kind) << "\n"
    << "  element_compute: " << to_string(k.element_compute) << "\n"
    << "   element_scalar: " << to_string(k.element_scalar) << "\n"
    << "        element_A: " << to_string(k.element_A) << "\n"
    << "         layout_A: " << to_string(k.layout_A) << "\n"
    << "      transform_A: " << to_string(k.transform_A) << "\n"
    << "        element_B: " << to_string(k.element_B) << "\n"
    << "         layout_B: " << to_string(k.layout_B) << "\n"
    << "      transform_B: " << to_string(k.transform_B) << "\n"
    << "        element_C: " << to_string(k.element_C) << "\n"
    << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for GemmFunctionalKey
struct GemmFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline
  static size_t rotl(size_t key, int shl) {
    return (key << shl) | (key >> (sizeof(key)*8 - shl));
  }

  inline
  size_t operator()(GemmFunctionalKey const &key) const {
    IntHash hash;

    return 
      rotl(hash(int(key.provider)), 1) ^ 
      rotl(hash(int(key.gemm_kind)), 2) ^ 
      rotl(hash(int(key.element_compute)), 3) ^
      rotl(hash(int(key.element_scalar)), 4) ^
      rotl(hash(int(key.element_A)), 5) ^
      rotl(hash(int(key.layout_A)), 6) ^
      rotl(hash(int(key.transform_A)), 7) ^
      rotl(hash(int(key.element_B)), 8) ^
      rotl(hash(int(key.layout_B)), 9) ^
      rotl(hash(int(key.transform_B)), 10) ^
      rotl(hash(int(key.element_C)), 11);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes a partial ordering to search for GEMM operators
struct GemmPreferenceKey {

  int compute_capability;
  int alignment;

  //
  // Methods
  //

  GemmPreferenceKey(): compute_capability(), alignment() { }

  GemmPreferenceKey(int cc, int alignment): compute_capability(cc), alignment(alignment) { }

  bool operator<(GemmPreferenceKey const &rhs) const {
    return (compute_capability < rhs.compute_capability) || 
      ((compute_capability == rhs.compute_capability) && (alignment < rhs.alignment));
  }

  bool operator==(GemmPreferenceKey const &rhs) const {
    return compute_capability == rhs.compute_capability;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps minimum compute capability onto a vector of possible operations
using GemmOperationVectorMap = std::map<
  GemmPreferenceKey,
  std::vector<Operation const *>
>;

/// Maps a GemmFunctionalKey onto a vector of Operation * objects expected to be of kind kGemm
using GemmOperationFunctionalMap = std::unordered_map<
  GemmFunctionalKey,
  GemmOperationVectorMap,
  GemmFunctionalKeyHasher
>;
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for Conv Functional Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tuple uniquely identifying conv2d functional behavior
struct ConvFunctionalKey {
  library::Provider provider;
  library::ConvKind conv_kind;
  library::NumericTypeID element_A;
  library::LayoutTypeID layout_A;
  library::NumericTypeID element_B;
  library::LayoutTypeID layout_B;
  library::NumericTypeID element_C;
  library::LayoutTypeID layout_C;
  library::NumericTypeID element_accumulator;
  library::NumericTypeID element_compute;


  //
  // Methods
  //

  inline
  ConvFunctionalKey(
    library::Provider provider = library::Provider::kInvalid,
    library::ConvKind conv_kind = library::ConvKind::kFprop,
    library::NumericTypeID element_A = library::NumericTypeID::kF16,
    library::LayoutTypeID layout_A = library::LayoutTypeID::kTensorNHWC,
    library::NumericTypeID element_B = library::NumericTypeID::kF16,
    library::LayoutTypeID layout_B = library::LayoutTypeID::kTensorNHWC,
    library::NumericTypeID element_C = library::NumericTypeID::kF16,
    library::LayoutTypeID layout_C = library::LayoutTypeID::kTensorNHWC,
    library::NumericTypeID element_accumulator = library::NumericTypeID::kF32,
    library::NumericTypeID element_compute = library::NumericTypeID::kF32
  ):
    provider(provider),
    conv_kind(conv_kind),
    element_A(element_A),
    layout_A(layout_A),
    element_B(element_B),
    layout_B(layout_B),
    element_C(element_C),
    layout_C(layout_C),
    element_accumulator(element_accumulator),
    element_compute(element_compute)
  { } 

  inline 
  bool operator==(ConvFunctionalKey const &rhs) const {
    return
      (provider == rhs.provider) &&
      (conv_kind == rhs.conv_kind) &&
      (element_A == rhs.element_A) &&
      (layout_A == rhs.layout_A) &&
      (element_B == rhs.element_B) &&
      (layout_B == rhs.layout_B) &&
      (element_C == rhs.element_C) &&
      (layout_C == rhs.layout_C) &&
      (element_accumulator == rhs.element_accumulator) &&
      (element_compute == rhs.element_compute);
  }

  inline 
  bool operator!=(ConvFunctionalKey const &rhs) const {
    return !(*this == rhs);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////
inline
std::ostream& operator<< (std::ostream& out, const cutlass::library::ConvFunctionalKey& key) {
    out << "{\n"
      << "provider: " << to_string(key.provider) << std::endl
      << "conv_kind: " << to_string(key.conv_kind) << std::endl
      << "element_A: " << to_string(key.element_A) << std::endl
      << "layout_A: " << to_string(key.layout_A) << std::endl
      << "element_B: " << to_string(key.element_B) << std::endl
      << "layout_B: " << to_string(key.layout_B) << std::endl
      << "element_C: " << to_string(key.element_C) << std::endl
      << "layout_C: " << to_string(key.layout_C) << std::endl
      << "element_accumulator: " << to_string(key.element_accumulator) << std::endl
      << "element_compute: " << to_string(key.element_compute) << std::endl
      << "}";
  
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
struct ConvFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline
  static size_t rotl(size_t key, int shl) {
    return (key << shl) | (key >> (sizeof(key)*8 - shl));
  }

  inline
  size_t operator()(ConvFunctionalKey const &key) const {
    IntHash hash;

    return 
      rotl(hash(int(key.provider)), 1) ^
      rotl(hash(int(key.conv_kind)), 2) ^
      rotl(hash(int(key.element_A)), 3) ^
      rotl(hash(int(key.layout_A)), 4) ^
      rotl(hash(int(key.element_B)), 5) ^
      rotl(hash(int(key.layout_B)), 6) ^
      rotl(hash(int(key.element_C)), 7) ^
      rotl(hash(int(key.layout_C)), 8) ^
      rotl(hash(int(key.element_accumulator)), 9) ^
      rotl(hash(int(key.element_compute)), 10);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes a partial ordering to search for Conv2d operators
struct ConvPreferenceKey {

  int compute_capability;
  IteratorAlgorithmID iterator_algorithm;


  //
  // Methods
  //

  ConvPreferenceKey(): compute_capability(), iterator_algorithm() { }

  ConvPreferenceKey(int cc, IteratorAlgorithmID iterator_algorithm): 
    compute_capability(cc), iterator_algorithm(iterator_algorithm) { }

  bool operator<(ConvPreferenceKey const &rhs) const {
    return (compute_capability < rhs.compute_capability) || 
      ((compute_capability == rhs.compute_capability) && (iterator_algorithm < rhs.iterator_algorithm));
  }

  bool operator==(ConvPreferenceKey const &rhs) const {
    return (compute_capability == rhs.compute_capability) &&
          (iterator_algorithm == rhs.iterator_algorithm);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps minimum compute capability onto a vector of possible operations
using ConvOperationVectorMap = std::map<
  ConvPreferenceKey,
  std::vector<Operation const *>
>;

/// Maps a GemmFunctionalKey onto a vector of Operation * objects expected to be of kind kGemm
using ConvOperationFunctionalMap = std::unordered_map<
  ConvFunctionalKey,
  ConvOperationVectorMap,
  ConvFunctionalKeyHasher
>;
/////////////////////////////////////////////////////////////////////////////////////////////////


/// Tuple uniquely identifying conv2d functional behavior
struct ReductionFunctionalKey {
  library::Provider provider;
  library::NumericTypeID element_workspace;
  library::NumericTypeID element_accumulator;
  library::NumericTypeID element_output;
  library::NumericTypeID element_compute;
  library::MathOperationID reduce_math_op;
  library::EpilogueKind epilogue_math_op;


  //
  // Methods
  //

  inline
  ReductionFunctionalKey(
    library::Provider provider = library::Provider::kInvalid,
    library::NumericTypeID element_workspace = library::NumericTypeID::kF16,
    library::NumericTypeID element_accumulator = library::NumericTypeID::kF32,
    library::NumericTypeID element_output = library::NumericTypeID::kF16,
    library::NumericTypeID element_compute = library::NumericTypeID::kF32,
    library::MathOperationID reduce_math_op = library::MathOperationID::kAdd,
    library::EpilogueKind epilogue_math_op = library::EpilogueKind::kLinearCombination
  ):
    provider(provider),
    element_workspace(element_workspace),
    element_accumulator(element_accumulator),
    element_output(element_output),
    element_compute(element_compute),
    reduce_math_op(reduce_math_op),
    epilogue_math_op(epilogue_math_op)
  { } 

  inline 
  bool operator==(ReductionFunctionalKey const &rhs) const {
    return
      (provider == rhs.provider) &&
      (element_workspace == rhs.element_workspace) &&
      (element_accumulator == rhs.element_accumulator) &&
      (element_output == rhs.element_output) &&
      (element_compute == rhs.element_compute) &&
      (reduce_math_op == rhs.reduce_math_op) &&
      (epilogue_math_op == rhs.epilogue_math_op);
  }

  inline 
  bool operator!=(ReductionFunctionalKey const &rhs) const {
    return !(*this == rhs);
  }
};


struct ReductionFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline
  static size_t rotl(size_t key, int shl) {
    return (key << shl) | (key >> (sizeof(key)*8 - shl));
  }

  inline
  size_t operator()(ReductionFunctionalKey const &key) const {
    IntHash hash;

    return 
      rotl(hash(int(key.provider)), 1) ^
      rotl(hash(int(key.element_workspace)), 2) ^
      rotl(hash(int(key.element_accumulator)), 3) ^
      rotl(hash(int(key.element_output)), 4) ^
      rotl(hash(int(key.element_compute)), 5) ^
      rotl(hash(int(key.reduce_math_op)), 6) ^
      rotl(hash(int(key.epilogue_math_op)), 7);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

inline
std::ostream& operator<< (std::ostream& out, const ReductionFunctionalKey& key) {
    out << "{\n"
      << "provider: " << library::to_string(key.provider) << std::endl
      << "element_workspace   : " << library::to_string(key.element_workspace) << std::endl
      << "element_accumulator : " << library::to_string(key.element_accumulator) << std::endl
      << "element_output      : " << library::to_string(key.element_output) << std::endl
      << "element_compute     : " << library::to_string(key.element_compute) << std::endl
      << "}";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// ReductionOperationFunctionalMap has NO preference key and a single instance per functional key
// i.e. only one tile size configuration per functional key
using ReductionOperationFunctionalMap = std::unordered_map<
  ReductionFunctionalKey,
  library::Operation const *,
  ReductionFunctionalKeyHasher
>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Table of cutlass::library::Operation instances
class OperationTable {
public:

  /// Map of all operations of type kGemm 
  // provider (kCUTLASS)
  GemmOperationFunctionalMap gemm_operations;

  /// Map of all operations of type kConv2d 
  // provider (kCUTLASS, kReferenceHost, kReferenceDevice)
  ConvOperationFunctionalMap conv2d_operations;

  /// Map of all operations of type kConv3d 
  // provider (kCUTLASS, kReferenceHost, kReferenceDevice)
  ConvOperationFunctionalMap conv3d_operations;

  /// Map of all operations of type kConv2d 
  // provider (kCUTLASS)
  ReductionOperationFunctionalMap reduction_operations;

public:

  void append(Manifest const &manifest);

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream & operator<<(std::ostream &out, cutlass::library::GemmFunctionalKey const &k);
