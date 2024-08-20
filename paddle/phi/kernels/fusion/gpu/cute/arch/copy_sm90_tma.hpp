/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_1D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes"
        " [%0], [%1, {%3}], [%2];"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_2D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "r"(crd0),
          "r"(crd1)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_3D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_4D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_5D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3),
          "r"(crd4)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
    return SM90_TMA_LOAD_1D::copy(desc_ptr, smem_mbar, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
    return SM90_TMA_LOAD_2D::copy(desc_ptr, smem_mbar, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
    return SM90_TMA_LOAD_3D::copy(
        desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
    return SM90_TMA_LOAD_4D::copy(
        desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
    return SM90_TMA_LOAD_5D::copy(
        desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD_MULTICAST: Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_1D_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.multicast::cluster"
        " [%0], [%1, {%4}], [%2], %3;"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "h"(multicast_mask),
          "r"(crd0)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_2D_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.multicast::cluster"
        " [%0], [%1, {%4, %5}], [%2], %3;"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "h"(multicast_mask),
          "r"(crd0),
          "r"(crd1)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_3D_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6}], [%2], %3;"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "h"(multicast_mask),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_4D_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6, %7}], [%2], %3;"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "h"(multicast_mask),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_5D_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6, %7, %8}], [%2], %3;"
        :
        : "r"(smem_int_ptr),
          "l"(gmem_int_desc),
          "r"(smem_int_mbar),
          "h"(multicast_mask),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3),
          "r"(crd4)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
    return SM90_TMA_LOAD_1D_MULTICAST::copy(
        desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
    return SM90_TMA_LOAD_2D_MULTICAST::copy(
        desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
    return SM90_TMA_LOAD_3D_MULTICAST::copy(
        desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
    return SM90_TMA_LOAD_4D_MULTICAST::copy(
        desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    uint64_t& smem_mbar,
                                    uint16_t multicast_mask,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
    return SM90_TMA_LOAD_5D_MULTICAST::copy(desc_ptr,
                                            smem_mbar,
                                            multicast_mask,
                                            smem_ptr,
                                            crd0,
                                            crd1,
                                            crd2,
                                            crd3,
                                            crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_STORE : Initiates a TMA copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_STORE_1D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], "
        "[%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_2D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
        "[%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_3D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, "
        "%4}], [%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_4D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, "
        "%4, %5}], [%1];"
        :
        : "l"(gmem_int_desc),
          "r"(smem_int_ptr),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_5D {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, "
        "%4, %5, %6}], [%1];"
        :
        : "l"(gmem_int_desc),
          "r"(smem_int_ptr),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3),
          "r"(crd4)
        : "memory");
#else
    CUTE_RUNTIME_ASSERT(
        "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE {
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0) {
    return SM90_TMA_STORE_1D::copy(desc_ptr, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1) {
    return SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2) {
    return SM90_TMA_STORE_3D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3) {
    return SM90_TMA_STORE_4D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void copy(void const* const desc_ptr,
                                    void const* const smem_ptr,
                                    int32_t const& crd0,
                                    int32_t const& crd1,
                                    int32_t const& crd2,
                                    int32_t const& crd3,
                                    int32_t const& crd4) {
    return SM90_TMA_STORE_5D::copy(
        desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

// Indicate arrival of warp issuing TMA_STORE
CUTE_HOST_DEVICE static void tma_store_arrive() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  asm volatile("cp.async.bulk.commit_group;");
#else
  CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Wait on prior N (Count) TMA_STORE instructions to complete
template <int Count>
CUTE_HOST_DEVICE static void tma_store_wait() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
#else
  CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // end namespace cute
