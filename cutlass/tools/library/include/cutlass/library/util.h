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
/*!
  \file

  \brief Utilities accompanying the CUTLASS library for interacting with Library types.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexical cast from string
template <typename T> T from_string(std::string const &);

/// Converts a Provider enumerant to a string
char const *to_string(Provider provider, bool pretty = false);

/// Parses a Provider enumerant from a string
template <> Provider from_string<Provider>(std::string const &str);

/// Converts a GemmKind enumerant to a string
char const *to_string(GemmKind type, bool pretty = false);

/// Converts a RankKKind enumerant to a string
char const *to_string(RankKKind type, bool pretty = false);

/// Converts a TrmmKind enumerant to a string
char const *to_string(TrmmKind type, bool pretty = false);

/// Converts a SymmKind enumerant to a string
char const *to_string(SymmKind type, bool pretty = false);

/// Converts a SideMode enumerant to a string
char const *to_string(SideMode type, bool pretty = false);

/// Converts a FillMode enumerant to a string
char const *to_string(FillMode type, bool pretty = false);

/// Converts a BlasMode enumerant to a string
char const *to_string(BlasMode type, bool pretty = false);

/// Converts a DiagType enumerant to a string
char const *to_string(DiagType type, bool pretty = false);

/// Converts a NumericType enumerant to a string
char const *to_string(OperationKind type, bool pretty = false);

/// Parses a NumericType enumerant from a string
template <> OperationKind from_string<OperationKind>(std::string const &str);

/// Converts a NumericType enumerant to a string
char const *to_string(NumericTypeID type, bool pretty = false);

/// Parses a NumericType enumerant from a string
template <> NumericTypeID from_string<NumericTypeID>(std::string const &str);

/// Returns the size of a data type in bits
int sizeof_bits(NumericTypeID type);

/// Returns true if the numeric type is a complex data type or false if real-valued.
bool is_complex_type(NumericTypeID type);

/// Returns the real-valued type underlying a type (only different from 'type' if complex)
NumericTypeID get_real_type(NumericTypeID type);

/// Returns true if numeric type is integer
bool is_integer_type(NumericTypeID type);

/// Returns true if numeric type is signed
bool is_signed_type(NumericTypeID type);

/// Returns true if numeric type is a signed integer
bool is_signed_integer(NumericTypeID type);

/// returns true if numeric type is an unsigned integer
bool is_unsigned_integer(NumericTypeID type);

/// Returns true if numeric type is floating-point type
bool is_float_type(NumericTypeID type);

/// To string method for cutlass::Status
char const *to_string(Status status, bool pretty = false);

/// Converts a LayoutTypeID enumerant to a string
char const *to_string(LayoutTypeID layout, bool pretty = false);

/// Parses a LayoutType enumerant from a string
template <> LayoutTypeID from_string<LayoutTypeID>(std::string const &str);

/// Returns the rank of a layout's stride base on the LayoutTypeID
int get_layout_stride_rank(LayoutTypeID layout_id);

/// Converts a OpcodeClassID enumerant to a string
char const *to_string(OpcodeClassID type, bool pretty = false);

/// Converts a OpcodeClassID enumerant from a string
template <>
OpcodeClassID from_string<OpcodeClassID>(std::string const &str);

/// Converts a ComplexTransform enumerant to a string
char const *to_string(ComplexTransform type, bool pretty = false);

/// Converts a ComplexTransform enumerant from a string
template <>
ComplexTransform from_string<ComplexTransform>(std::string const &str);


/// Converts a SplitKMode enumerant to a string
char const *to_string(SplitKMode split_k_mode, bool pretty = false);

/// Converts a SplitKMode enumerant from a string
template <>
SplitKMode from_string<SplitKMode>(std::string const &str);

/// Converts a ConvModeID enumerant to a string
char const *to_string(ConvModeID type, bool pretty = false);

/// Converts a ConvModeID enumerant from a string
template <>
ConvModeID from_string<ConvModeID>(std::string const &str);

/// Converts a IteratorAlgorithmID enumerant to a string
char const *to_string(IteratorAlgorithmID type, bool pretty = false);

/// Converts a IteratorAlgorithmID enumerant from a string
template <>
IteratorAlgorithmID from_string<IteratorAlgorithmID>(std::string const &str);

/// Converts a ConvKind enumerant to a string
char const *to_string(ConvKind type, bool pretty = false);

/// Converts a ConvKind enumerant from a string
template <>
ConvKind from_string<ConvKind>(std::string const &str);

/// Lexical cast from int64_t to string
std::string lexical_cast(int64_t int_value);

/// Lexical cast a string to a byte array. Returns true if cast is successful or false if invalid.
bool lexical_cast(std::vector<uint8_t> &bytes, NumericTypeID type, std::string const &str);

/// Lexical cast TO a string FROM a byte array. Returns true if cast is successful or false if invalid.
std::string lexical_cast(std::vector<uint8_t> &bytes, NumericTypeID type);

/// Casts from a signed int64 to the destination type. Returns true if successful.
bool cast_from_int64(std::vector<uint8_t> &bytes, NumericTypeID type, int64_t src);

/// Casts from an unsigned int64 to the destination type. Returns true if successful.
bool cast_from_uint64(std::vector<uint8_t> &bytes, NumericTypeID type, uint64_t src);

/// Casts from a real value represented as a double to the destination type. Returns true if successful.
bool cast_from_double(std::vector<uint8_t> &bytes, NumericTypeID type, double src);

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

