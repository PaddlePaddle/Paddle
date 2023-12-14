// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <cstddef>
#include <type_traits>

template <typename T, std::size_t Alignment>
struct AlignedType {
    alignas(Alignment) T data;

    // Default constructor
    AlignedType() = default;

    // Constructor to initialize with a value of type T
    explicit AlignedType(const T &value) : data(value) {}

    // Conversion operator to convert AlignedType to T
    operator T() const { return data; }

    // Overload the assignment operator to assign a value of type T
    AlignedType &operator=(const T &value) {
        data = value;
        return *this;
    }
};