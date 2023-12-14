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

#include <cstdint>
#include <type_traits>

template <typename T1, typename T2>
inline T1 bit_cast(const T2 &u) {
    static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
    static_assert(std::is_trivial<T1>::value, "T1 must be trivially copyable.");
    static_assert(std::is_trivial<T2>::value, "T2 must be trivially copyable.");

    T1 t;
    uint8_t *t_ptr = reinterpret_cast<uint8_t *>(&t);
    const uint8_t *u_ptr = reinterpret_cast<const uint8_t *>(&u);
    for (size_t i = 0; i < sizeof(T2); i++)
        t_ptr[i] = u_ptr[i];
    return t;
}
