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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iostream>

class uint4x2_t {
public:
    uint4x2_t() = default;
    uint4x2_t(uint8_t v1, uint8_t v2);
    uint4x2_t(uint8_t v1);

    uint4x2_t &operator=(const uint4x2_t& other);
    bool operator!=(const uint4x2_t& other) const;
    uint8_t get_v1() const;
    uint8_t get_v2() const;
    void print() const;

private:
    uint8_t raw_bits_;
};

static_assert(sizeof(uint4x2_t) == 1, "uint4x2_t must be 1 bytes");

inline uint4x2_t::uint4x2_t(uint8_t v1, uint8_t v2) {
    // In little-endian mode, the low-order byte is stored at
    // the low address end of memory. Merge v1 and v2.
    this->raw_bits_ = (v1 & 0x0F) | ((v2 & 0x0F) << 4);
}

inline uint4x2_t::uint4x2_t(uint8_t v1) {
    this->raw_bits_ = v1 & 0x0F;
}

inline uint4x2_t& uint4x2_t::operator=(const uint4x2_t& other) {
    if (this != &other) {
        raw_bits_ = other.raw_bits_;
    }

    return *this;
}

inline bool uint4x2_t::operator!=(const uint4x2_t& other) const {
    return raw_bits_ != other.raw_bits_;
}

inline uint8_t uint4x2_t::get_v1() const {
    return raw_bits_ & 0x0F;
}

inline uint8_t uint4x2_t::get_v2() const {
    return (raw_bits_ >> 4) & 0x0F;
}

inline void uint4x2_t::print() const {
    printf("uint4x2: 0x%x %d %d\n", raw_bits_, get_v1(), get_v2());
}
