/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
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

#pragma once

#include "gemm.h"
#include "utils.h"

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // The number of rows in the 2D shared memory buffer.
    int M_,
    // The number of cols.
    int N_,
    // The size in bits of each element.
    int BITS_PER_ELEMENT_,
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 16,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0,
    // The number of rows that are used for the XOR swizzling to allow fast
    // STS/LDS.
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast
    // STS/LDS.
    int COLS_PER_XOR_PATTERN_ = 1,
    // Use or not predicates
    bool USE_PREDICATES_ = true>
struct Smem_tile_without_skews {
  // The size in bits of each element.
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };
  // The size in bytes of a single STS.
  enum { BYTES_PER_STS = BYTES_PER_STS_ };
  // The number of elements per STS.
  enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
  // To support arbitrary N, we pad some values to a power-of-2.
  enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
  // The number of bytes per row without packing of rows.
  enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
  // The number of bytes per row -- we want at least 128B per row.
  enum { BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
  // The number of rows in shared memory (two rows may be packed into a single
  // one).
  enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW };

  // The number of threads per row.
  enum { THREADS_PER_ROW_UNBOUNDED = BYTES_PER_ROW / BYTES_PER_STS };
  // The number of threads per row.
  enum {
    THREADS_PER_ROW =
        Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_UNBOUNDED>::VALUE
  };

  // The number of STS per row.
  enum { STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS };
  // It must be at least one.
  static_assert(STS_PER_ROW >= 1, "");
  // The number of rows written with a single STS.
  enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
  // Make sure we write to at least one row per STS. Thanks Dr. Obvious ;)
  static_assert(ROWS_PER_STS >= 1, "");
  // The number of STS needed to store all rows.
  enum { STS_PER_COL = Div_up<ROWS, ROWS_PER_STS>::VALUE };
  // The number of STS in total.
  enum { STS = STS_PER_COL * STS_PER_ROW };

  // The size of one buffer in bytes in shared memory.
  enum { BYTES_PER_BUFFER = STS * BYTES_PER_STS * Cta_tile::THREADS_PER_CTA };
  // The number of buffers.
  enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
  // The size in bytes of total buffers.
  enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
  // The boundary for smem_read_offset and smem_write_offset increment.
  enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

  // Do we enable the LDS.128 fast path?
  enum { ENABLE_LDS_FAST_PATH = ENABLE_LDS_FAST_PATH_ };
  static_assert(ENABLE_LDS_FAST_PATH == 0);
  // The number of rows that are used for the XOR swizzling to allow fast
  // STS/LDS.
  enum { ROWS_PER_XOR_PATTERN = ROWS_PER_XOR_PATTERN_ };
  // The number of cols that are used for the XOR swizzling to allow fast
  // STS/LDS.
  enum { COLS_PER_XOR_PATTERN = COLS_PER_XOR_PATTERN_ * 16 / BYTES_PER_STS };
  // Use or not predicates
  enum { USE_PREDICATES = USE_PREDICATES_ };

  // The type of elements that are stored in shared memory by each thread.
  using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

  // Ctor.
  inline __device__ Smem_tile_without_skews(void *smem, int tidx)
      : smem_(__nvvm_get_smem_pointer(smem)) {
    // The row written by a thread. See doc/mma_smem_layout.xlsx.
    int smem_write_row = tidx / THREADS_PER_ROW;

    // The XOR pattern.
    int smem_write_xor =
        smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN;
    // Compute the column and apply the XOR pattern.
    int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;

    // The offset.
    this->smem_write_offset_ =
        smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;

    // TODO: Why not merge it with the read offset?
    this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
  }

  // Compute the store pointers.
  template <int N>
  inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
#pragma unroll
    for (int ii = 0; ii < N; ++ii) {
      // Decompose the STS into row/col.
      int row = ii / STS_PER_ROW;
      int col = ii % STS_PER_ROW;

      // Assemble the offset.
      int offset = smem_write_offset_ + row * ROWS_PER_STS * BYTES_PER_ROW;

      // Take the column into account.
      if (STS_PER_ROW > 1) {
        offset += col * THREADS_PER_ROW * BYTES_PER_STS;
      }

      // Apply the XOR pattern if needed.
      if (ROWS_PER_STS < ROWS_PER_XOR_PATTERN) {
        const int m = row * ROWS_PER_STS % ROWS_PER_XOR_PATTERN;
        offset ^= m * COLS_PER_XOR_PATTERN * BYTES_PER_STS;
      }

      // Assemble the final pointer :)
      ptrs[ii] = smem_ + offset + smem_write_buffer_;
    }
  }

  inline __device__ void debug_reset() {
    for (int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
      for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < BYTES_PER_ROW; col += 4) {
          if (threadIdx.x == 0) {
            uint32_t val = 0x0;
            sts(val, smem_ + row * BYTES_PER_ROW + col + buffer);
          }
        }
      }
    }
  }

  // Print the content of the tile (only for debug ;)).
  inline __device__ void debug_print() const {
    for (int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
      for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < BYTES_PER_ROW; col += 4) {
          if (threadIdx.x == 0) {
            uint32_t val;
            lds(val, smem_ + row * BYTES_PER_ROW + col + buffer);
            printf(
                "block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, "
                "byte=%4d)=0x%08x\n",
                blockIdx.x, blockIdx.y, blockIdx.z, smem_, buffer, row, col,
                val);
          }
        }
      }
    }
  }

  // Move the read offset to next buffer.
  inline __device__ void move_to_next_read_buffer() {
    if (BUFFERS_PER_TILE > 1 &&
        smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY) {
      this->smem_read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
    } else if (BUFFERS_PER_TILE > 1) {
      this->smem_read_buffer_ += BYTES_PER_BUFFER;
    }
  }

  // Move the read offset to next buffer. TODO: Remove this member function!!!
  inline __device__ void move_next_read_buffer() {
    this->move_to_next_read_buffer();
  }

  // Move the read offset to next N buffer (circular-buffer).
  inline __device__ void move_to_next_read_buffer(int N) {
    if (BUFFERS_PER_TILE > 1) {
      this->smem_read_buffer_ += N * BYTES_PER_BUFFER;
      this->smem_read_buffer_ -=
          smem_read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
    }
  }

  // Move the read offset to next N buffer (circular-buffer). TODO: Remove this
  // member function!!!
  inline __device__ void move_next_read_buffer(int N) {
    this->move_to_next_read_buffer(N);
  }

  // Move the write offset to next buffer.
  inline __device__ void move_to_next_write_buffer() {
    if (BUFFERS_PER_TILE > 1 &&
        smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY) {
      this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
    } else if (BUFFERS_PER_TILE > 1) {
      this->smem_write_buffer_ += BYTES_PER_BUFFER;
    }
  }

  // Move the write offset to next buffer. TODO: Remove that member function!
  inline __device__ void move_next_write_buffer() {
    this->move_to_next_write_buffer();
  }

  // Move the read offset.
  inline __device__ void move_read_offset(int delta) {
    this->smem_read_offset_ += delta;
  }

  // Move the write offset.
  inline __device__ void move_write_offset(int delta) {
    this->smem_write_offset_ += delta;
  }

  // Store to the tile in shared memory.
  template <int N>
  inline __device__ void store(const Store_type (&data)[N], uint64_t = 0) {
    uint32_t smem_ptrs[N];
    this->compute_store_pointers(smem_ptrs);
    sts(smem_ptrs, data);
  }

  // Store to the tile in shared memory.
  template <int N, int M>
  inline __device__ void store(const Store_type (&data)[N],
                               uint32_t (&preds)[M], uint64_t = 0) {
    uint32_t smem_ptrs[N];
    this->compute_store_pointers(smem_ptrs);
    sts(smem_ptrs, data, preds);
  }

  // Store to the tile in shared memory.
  template <int N>
  inline __device__ void store(const Store_type (&data)[N], uint32_t preds,
                               uint64_t = 0) {
    this->store(data, preds);
  }

  // Store to the tile in shared memory.
  template <int N>
  inline __device__ void store(const void *(&gmem_ptrs)[N], uint32_t preds,
                               uint64_t = 0) {
    uint32_t tmp[1] = {preds};
    this->store(gmem_ptrs, tmp);
  }

  // The shared memory pointer.
  uint32_t smem_;
  // The read offset. Reserve 4 offsets if needed.
  int smem_read_offset_;
  // The write offset.
  int smem_write_offset_;
  // The buffer base offset for read.
  int smem_read_buffer_;
  // The buffer base offset for write.
  int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true>
struct Smem_tile_a {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K, int MMAS_K_WITH_PADDING>
struct Compute_reset_mask {
  // The potential mask.
  enum { HALF = MMAS_K_WITH_PADDING / 2 };
  // The remainder.
  enum { MOD = MMAS_K % HALF };
  // The final value.
  enum {
    VALUE = (MMAS_K == MOD ? 0 : HALF) | Compute_reset_mask<MOD, HALF>::VALUE
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K_WITH_PADDING>
struct Compute_reset_mask<0, MMAS_K_WITH_PADDING> {
  enum { VALUE = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K>
struct Compute_reset_mask<MMAS_K, MMAS_K> {
  enum { VALUE = MMAS_K - 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Rows_per_xor_pattern_a {
  // The size in bits.
  enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_A };
  // The number of rows.
  enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Rows_per_xor_pattern_row_a : public Rows_per_xor_pattern_a<N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_row_a<Cta_tile::K>::VALUE>
struct Smem_tile_row_a
    : public Smem_tile_without_skews<
          Cta_tile, Cta_tile::M, Cta_tile::K, fmha::BITS_PER_ELEMENT_A,
          BYTES_PER_STS, BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1> {
  // The MMA tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  // The base class.
  using Base =
      Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K,
                              fmha::BITS_PER_ELEMENT_A, BYTES_PER_STS,
                              BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
  // The fragment.
  using Fragment = Fragment_a<Row>;

  // When we use padding to reach a power of two, special care has to be taken.
  using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Cta_tile>;
  // The number of MMAs.
  using Mma_tile_with_padding = fmha::Hmma_tile<Cta_tile_with_padding>;

  // The size of a single LDS in bytes.
  enum { BYTES_PER_LDS = 16 };

  // Ctor.
  inline __device__ Smem_tile_row_a(void *smem, int tidx) : Base(smem, tidx) {
    // For documentation on the layout, see doc/mma_smem_layout.xlsx.

    // The number of warps.
    const int WARPS_M = Cta_tile::WARPS_M;
    const int WARPS_N = Cta_tile::WARPS_N;
    const int WARPS_K = Cta_tile::WARPS_K;

    static_assert(WARPS_M == 1);
    static_assert(WARPS_N == 4 || WARPS_N == 8);
    static_assert(WARPS_K == 1);
    static_assert(Base::ROWS_PER_XOR_PATTERN == 8);

    // The row and column read by the thread.
    int smem_read_row = (tidx & 0x0f);
    int smem_read_col = (tidx & 0x07);
    smem_read_col ^= (tidx & 0x10) / 16;

    // The shared memory offset.
    this->smem_read_offset_ =
        smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
  }

  // Rewind smem_read_offset for last LDS phase in main loop.
  inline __device__ void reverse_smem_read_offset(int ki = 0) {
    // Undo the pointer increment for the next ni.
    // Should match the load function below for ki = 0.
    if (Mma_tile_with_padding::MMAS_K >= 2) {
      this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
    }
  }

  // Load from shared memory.
  inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki) {
#pragma unroll
    for (int mi = 0; mi < Mma_tile::MMAS_M; ++mi) {
      // Jump by as many matrix rows as needed (a row in smem may pack multiple
      // matrix rows).
      int offset =
          mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

      // Load using LDSM.M88.4.
      uint4 tmp;
      ldsm(tmp, this->smem_ + this->smem_read_offset_ +
                    this->smem_read_buffer_ + offset);

      // Store the value into the fragment.
      a[mi].reg(0) = tmp.x;
      a[mi].reg(1) = tmp.y;
      a[mi].reg(2) = tmp.z;
      a[mi].reg(3) = tmp.w;
    }

    // Move the offset to the next possition. See doc/mma_smem_layout.xlsx.
    static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
    if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15) {
      this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7) {
      this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3) {
      this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1) {
      this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 2) {
      this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
    }
  }

  // Reset the read offset.
  inline __device__ void reset_read_offset() {
    // The number of MMAs in the K dimension.
    enum { MMAS_K = Mma_tile::MMAS_K };
    // The number of MMAs in the K dimension when we include padding.
    enum { MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K };
    // Assemble the mask.
    enum { MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE };

    // Reset the read offset.
    this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_row_a<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE> {
  // The base class.
  using Base = Smem_tile_row_a<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

  // Ctor.
  inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true>
struct Smem_tile_b {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Rows_per_xor_pattern_b {
  // The size in bits.
  enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_B };
  // The number of rows.
  enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Rows_per_xor_pattern_col_b : public Rows_per_xor_pattern_b<N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_col_b<Cta_tile::K>::VALUE>
struct Smem_tile_col_b
    : public Smem_tile_without_skews<
          Cta_tile, Cta_tile::N, Cta_tile::K, fmha::BITS_PER_ELEMENT_B,
          BYTES_PER_STS, BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1> {
  // The MMA tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  // The base class.
  using Base =
      Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K,
                              fmha::BITS_PER_ELEMENT_B, BYTES_PER_STS,
                              BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
  // The fragment.
  using Fragment = Fragment_b<Col>;

  // When we use padding to reach a power of two, special care has to be taken.
  using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Cta_tile>;
  // The number of MMAs.
  using Mma_tile_with_padding = fmha::Hmma_tile<Cta_tile_with_padding>;

  // The size of a single LDS in bytes.
  enum { BYTES_PER_LDS = 16 };

  // The number of STS per thread
  enum {
    STS_PER_THREAD_ =
        Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA
  };
  // The number of STS per thread must be at least 1.
  enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

  // Ctor.
  inline __device__ Smem_tile_col_b(void *smem, int tidx) : Base(smem, tidx) {
    // For documentation on the layout, see doc/mma_smem_layout.xlsx.

    // The number of warps.
    const int WARPS_M = Cta_tile::WARPS_M;
    const int WARPS_N = Cta_tile::WARPS_N;
    const int WARPS_K = Cta_tile::WARPS_K;
    static_assert(Base::ROWS_PER_XOR_PATTERN == 8);
    static_assert(WARPS_M == 1);
    static_assert(WARPS_N == 4 || WARPS_N == 8);
    static_assert(WARPS_K == 1);

    // The masks to select the warps.
    const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

    // The divisor for the warps.
    const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

    // The row and column read by the thread.
    int smem_read_row =
        (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA +
        (tidx & 0x07) + (tidx & 0x10) / 2;
    int smem_read_col = (tidx & 0x07);
    smem_read_col ^= (tidx & 0x08) / 8;
    // The shared memory offset.
    this->smem_read_offset_ =
        smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
  }

  // Rewind smem_read_offset for last LDS phase in main loop.
  inline __device__ void reverse_smem_read_offset(int ki = 0) {
    // Undo the pointer increment for the next ni.
    // Should match the load function below for ki = 0.
    if (Mma_tile_with_padding::MMAS_K >= 2) {
      this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
    }
  }

  // Load from shared memory.
  inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
#pragma unroll
    for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni) {
      // Jump by as many matrix rows as needed (a row in smem may pack multiple
      // matrix rows).
      int offset =
          ni * Mma_tile::N_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

      // Load using LDSM.M88.4.
      uint4 tmp;
      ldsm(tmp, this->smem_ + this->smem_read_offset_ +
                    this->smem_read_buffer_ + offset);

      // Store the value into the fragment.
      b[ni].reg(0) = tmp.x;
      b[ni].reg(1) = tmp.y;
      b[ni].reg(2) = tmp.z;
      b[ni].reg(3) = tmp.w;
    }

    // Move the offset to the next possition. See doc/mma_smem_layout.xlsx.
    static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
    if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15) {
      this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7) {
      this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3) {
      this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1) {
      this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
    } else if (Mma_tile_with_padding::MMAS_K >= 2) {
      this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
    }
  }

  // Reset the read offset.
  inline __device__ void reset_read_offset() {
    // The number of MMAs in the K dimension.
    enum { MMAS_K = Mma_tile::MMAS_K };
    // The number of MMAs in the K dimension when we include padding.
    enum { MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K };
    // Assemble the mask.
    enum { MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE };

    // Reset the read offset.
    this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_col_b<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE> {
  // The base class.
  using Base = Smem_tile_col_b<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

  // Ctor.
  inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Rows_per_xor_pattern_row_b : public Rows_per_xor_pattern_b<N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_row_b<Cta_tile::N>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = 1>
struct Smem_tile_row_b
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N,
                                     fmha::BITS_PER_ELEMENT_B, BYTES_PER_STS,
                                     BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_,
                                     COLS_PER_XOR_PATTERN_> {
  // The MMA tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  // The base class.
  using Base =
      Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N,
                              fmha::BITS_PER_ELEMENT_B, BYTES_PER_STS,
                              BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_,
                              COLS_PER_XOR_PATTERN_>;
  // The fragment.
  using Fragment = Fragment_b<Row>;

  // Can we use LDSM? No if the data type is 32-bit large.
  enum { USE_LDSMT = fmha::BITS_PER_ELEMENT_B == 16 };
  // The size of a single LDS in bytes.
  enum { BYTES_PER_LDS = USE_LDSMT ? 16 : 4 };
  // The number of elements per LDS.
  enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / fmha::BITS_PER_ELEMENT_B };

  // The number of STS per thread
  enum {
    STS_PER_THREAD_ =
        Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA
  };
  // The number of STS per thread must be at least 1.
  enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

  // Ctor.
  inline __device__ Smem_tile_row_b(void *smem, int tidx) : Base(smem, tidx) {
    // The number of warps.
    const int WARPS_M = Cta_tile::WARPS_M;
    const int WARPS_N = Cta_tile::WARPS_N;
    const int WARPS_K = Cta_tile::WARPS_K;
    static_assert(WARPS_K == 1);
    static_assert(WARPS_M == 4 || WARPS_M == 8);
    static_assert(WARPS_N == 1);

    // The masks to select the warps.
    const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
    const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

    // The divisor for the warps.
    const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
    const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

    // The row/col read by the thread.
    int smem_read_row, smem_read_col;

    static_assert(USE_LDSMT);
    static_assert(Base::ROWS_PER_XOR_PATTERN == 8);

    smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile::MMAS_K * 16 +
                    (tidx & 0x07) + (tidx & 0x08);
    smem_read_col = (tidx & 0x07);
    smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 2 + (tidx & 0x10) / 16;

    // The shared memory offset.
    this->smem_read_offset_ =
        smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;

    // Fill zeroes for group conv
  }

  // Rewind smem_read_offset for last LDS phase in main loop.
  inline __device__ void reverse_smem_read_offset(int ki = 0) {
    // The size of each element in bits.
    const int BITS_PER_ELT = fmha::BITS_PER_ELEMENT_B;
    // The size in bytes of the data needed to compute an MMA per CTA.
    const int BYTES_PER_MMA_PER_CTA =
        Mma_tile::N_PER_MMA_PER_CTA * BITS_PER_ELT / 8;

#pragma unroll
    for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni) {
      // Undo the pointer increment for the next ni.
      // Should match the load function below for ki = 0.
      if (BYTES_PER_MMA_PER_CTA >= 128) {
        // Nothing to do!
      } else if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1) {
        this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
      } else if (BYTES_PER_MMA_PER_CTA == 64) {
        // Nothing to do!
      } else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 4) {
        this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
      } else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 2) {
        this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
      }
    }

    // Reset smem_read_offset for odd MMAS_N > 1 (npo2 kernels)
    if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1 &&
        Mma_tile::MMAS_N % 2 == 1) {
      this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
    }
  }

  // Load from shared memory.
  inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
    // The size of each element in bits.
    const int BITS_PER_ELT = fmha::BITS_PER_ELEMENT_B;
    // The size in bytes of the data needed to compute an MMA per CTA.
    const int BYTES_PER_MMA_PER_CTA =
        Mma_tile::N_PER_MMA_PER_CTA * BITS_PER_ELT / 8;

#pragma unroll
    for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni) {
      // Prepare the offset.
      int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
      if (BYTES_PER_MMA_PER_CTA == 32) {
        offset += this->smem_read_offset_;
      } else if (BYTES_PER_MMA_PER_CTA == 64) {
        offset +=
            this->smem_read_offset_ + (ni / 2) * BYTES_PER_MMA_PER_CTA * 2;
      } else {
        offset += this->smem_read_offset_ + (ni)*BYTES_PER_MMA_PER_CTA;
      }

      // Load the data using LDSM.MT88.2.
      uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
      uint4 tmp;
      if (USE_LDSMT) {
        ldsmt(tmp, ptr);
      } else {
        lds(tmp.x, (ptr) + 0 * Base::BYTES_PER_ROW);
        lds(tmp.y, (ptr) + 4 * Base::BYTES_PER_ROW);
        lds(tmp.z, (ptr ^ 32) + 0 * Base::BYTES_PER_ROW);
        lds(tmp.w, (ptr ^ 32) + 4 * Base::BYTES_PER_ROW);
      }

      // Store those values in the fragment.
      b[ni].reg(0) = tmp.x;
      b[ni].reg(1) = tmp.y;
      b[ni].reg(2) = tmp.z;
      b[ni].reg(3) = tmp.w;

      // Move the pointer for the next ni. I expect the compiler to not
      // recompute those.
      if (BYTES_PER_MMA_PER_CTA >= 128) {
        // Nothing to do!
      } else if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1) {
        this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
      } else if (BYTES_PER_MMA_PER_CTA == 64) {
        // Nothing to do!
      } else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 4) {
        this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
      } else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 2) {
        this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
      }
    }

    // Reset smem_read_offset for odd MMAS_N > 1 (npo2 kernels)
    if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1 &&
        Mma_tile::MMAS_N % 2 == 1) {
      this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_row_b<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE> {
  // The base class.
  using Base = Smem_tile_row_b<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

  // Ctor.
  inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_v
    : public fmha::Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N,
                                           16, 16, 1, 0, 8, 1> {
  // The base class.
  using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 16,
                                       16, 1, 0, 8, 1>;
  // The MMA tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  // The fragment.
  using Fragment = Fragment_b<fmha::Col>;

  // The size of a single LDS in bytes.
  enum { BYTES_PER_LDS = 16 };

  // Ctor.
  inline __device__ Smem_tile_v(void *smem, int tidx) : Base(smem, tidx) {
    // The row/col read by the thread.
    int read_row, read_col;

    static_assert(Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1 &&
                  (Cta_tile::WARPS_K == 4 || Cta_tile::WARPS_K == 8));

    read_row = (tidx & 0xe0) / 2 + (tidx & 0x0f);
    read_col = (tidx & 0x07);
    read_col ^= (tidx & 0x10) / 16;

    // The shared memory offset.
    this->smem_read_offset_ =
        read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
  }

  // Load from shared memory.
  inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
#pragma unroll
    for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni) {
      // Jump by 16 * #warps row.
      int row = ki * 16 * Cta_tile::WARPS_K;

      // Load the data using LDSM.MT88.2.
      uint4 tmp;
      fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ +
                           row * Base::BYTES_PER_ROW);
      b[ni].reg(0) = tmp.x;
      b[ni].reg(1) = tmp.y;
      b[ni].reg(2) = tmp.z;
      b[ni].reg(3) = tmp.w;

      // Move the pointer for the next ni. I expect the compiler to not
      // recompute those.
      if (Mma_tile::MMAS_N == 4) {
        this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
      } else {
        assert(false);  // Not implemented!
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o {
  // The MMA tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  // The accumulators.
  using Accumulator = fmha::Fragment_accumulator;
  // The accumulators.
  using Data_type = typename Accumulator::Data_type;

  // The size of each element.
  enum { BYTES_PER_ELEMENT = sizeof(Data_type) };
  // The size of each STS.
  enum { BYTES_PER_STS = 8 };
  // The size of each row in shared memory.
  enum { BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * BYTES_PER_ELEMENT };

  // The size of each LDS.
  enum { BYTES_PER_LDS = 16 };
  enum { THREADS_PER_ROW = 16 };

  // The number of rows.
  enum { ROWS = Cta_tile::M };
  // The number of "rows" to process per loop iteration (in the "epilogue").
  enum { ROWS_PER_LOOP = ROWS <= 64 ? ROWS : (int)Mma_tile::M_PER_MMA_PER_CTA };
  // The number of outer loops.
  enum { LOOPS = ROWS / ROWS_PER_LOOP };
  // Make sure it matches our expectations.
  static_assert(LOOPS == 1 || LOOPS == (int)Mma_tile::MMAS_M, "");

  // The number of rows loaded per LDS.
  enum { ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
  // Do we have to guard against partial writes/reads.
  enum { HAS_INCOMPLETE_LDS = ROWS_PER_LOOP % ROWS_PER_LDS != 0 };
  // The total number of LDS per loop.
  enum { LDS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_LDS>::VALUE };

  // The amount of shared memory.
  enum { BYTES_PER_TILE = ROWS_PER_LOOP * BYTES_PER_ROW };

  // The write pointer.
  uint32_t smem_write_, smem_read_;
  // Is the thread active for the last LDS of the series?
  int is_active_for_last_lds_;

  static_assert(BYTES_PER_ROW == 64 * 4 * Cta_tile::WARPS_K);
  static_assert(LOOPS == 1 || LOOPS == (int)Mma_tile::MMAS_M, "");

  // Ctor.
  inline __device__ Smem_tile_o(void *smem, int tidx) {
    // Get a 32-bit value for the shared memory address.
    uint32_t smem_ = __nvvm_get_smem_pointer(smem);

    static_assert(Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1 &&
                  (Cta_tile::WARPS_K == 4 || Cta_tile::WARPS_K == 8));

    int write_row = (tidx & 0x1c) / 4;
    int write_col = (tidx);

    // Assemble the write pointer.
    smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

    // The element read by each thread.
    int read_row = tidx / THREADS_PER_ROW;
    int read_col = tidx % THREADS_PER_ROW;

    // Take the XOR pattern into account for the column.
    read_col ^= 2 * (read_row & 0x7);

    // Assemble the read pointer.
    this->smem_read_ =
        smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

    // Is that thread active on the last LDS?
    if (HAS_INCOMPLETE_LDS) {
      this->is_active_for_last_lds_ =
          read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < Cta_tile::M;
    }
  }

  // Load the output fragments.
  inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const {
#pragma unroll
    for (int ii = 0; ii < LDS_PER_LOOP; ++ii) {
      // Load the elements before the reduction (split-K).
      uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
      for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj) {
        int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW +
                  jj * Cta_tile::N * BYTES_PER_ELEMENT;
        if (!HAS_INCOMPLETE_LDS ||
            (ii < LDS_PER_LOOP - 1 || this->is_active_for_last_lds_)) {
          fmha::lds(tmp[jj], this->smem_read_ + imm);
        }
      }

      // Perform the reduction.
      out[ii] = tmp[0];
#pragma unroll
      for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj) {
        out[ii] = fmha::fadd4(out[ii], tmp[jj]);
      }
    }
  }
  // Store the accumulators.
  template <int M, int N>
  inline __device__ void store(const Accumulator (&acc)[M][N], int mi) {
    enum { M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA };
#pragma unroll
    for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni) {
      // The number of MMAs that are stored per loop iteration.
      enum { MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS };

// Store 1st column of the different MMAs.
#pragma unroll
      for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj) {
        // Precompute the immediates to jump between rows.
        int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
        int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;
        uint2 tmp0, tmp1;
        tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(0);
        tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(1);

        tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(2);
        tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(3);

        // Store.
        fmha::sts(this->smem_write_ + row_0, tmp0);
        fmha::sts(this->smem_write_ + row_1, tmp1);
      }

      // Swizzle the write pointer using a XOR of 16B.
      this->smem_write_ ^= 32;

// Store 2nd column of the different MMAs.
#pragma unroll
      for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj) {
        // Precompute the immediates to jump between rows.
        int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
        int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

        uint2 tmp0, tmp1;
        tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(4);
        tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(5);

        tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(6);
        tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(7);
        // Store.
        fmha::sts(this->smem_write_ + row_0, tmp0);
        fmha::sts(this->smem_write_ + row_1, tmp1);
      }

      // Cancel the previous XOR of 1 + swizzle the write pointer using a XOR of
      // 32B or 64B.
      this->smem_write_ ^= (ni & 1) ? 7 * 32 : 3 * 32;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_mma {
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;
  using Fragment = fmha::Fragment_a<fmha::Col>;

  enum { COLS = Cta_tile::N };
  enum { BYTES_PER_ELT = 2 };
  enum { BYTES_PER_STS = 4 };
  enum { BYTES_PER_ROW = COLS * BYTES_PER_ELT };  // TODO
  enum { BYTES_PER_TILE = Cta_tile::M * BYTES_PER_ROW };

  enum { WARPS_M = Cta_tile::WARPS_M };
  enum { WARPS_N = Cta_tile::WARPS_N };
  enum { WARPS_K = Cta_tile::WARPS_K };

  static_assert(WARPS_K == 1);
  inline __device__ Smem_tile_mma(char *smem, int tidx) {
    smem_ = __nvvm_get_smem_pointer(smem);

    int write_col, write_row;
    static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8) ||
                  (WARPS_M == 4 || WARPS_N == 8) || WARPS_N == 1);
    if (WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8)) {
      write_row = (tidx & 0x1c) / 4;
      write_col = (tidx & 0xe0) / 4 + (tidx & 0x03);
    } else {
      write_row = (tidx & 0xe0) / 2 + (tidx & 0x1c) / 4;
      write_col = (tidx & 0x03);
    }
    write_col ^= (write_row & 0x07) * 4;

    write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;
  }

  template <int M, int N>
  inline __device__ void store(const uint4 (&regs)[M][N]) {
    static_assert(COLS == Cta_tile::N);
    for (int mi = 0; mi < M; mi++) {
      for (int ni = 0; ni < N; ni++) {
        size_t offset = write_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW +
                        ni * WARPS_N * 16 * BYTES_PER_ELT;
        fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
        fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
        offset ^= 4 * BYTES_PER_STS;
        fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
        fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
      }
    }
  }

  uint32_t smem_;
  uint32_t write_offset_;
  uint32_t warp_m;
  uint32_t warp_n;
  uint32_t lane;
};

template <typename Cta_tile, typename Base = Smem_tile_mma<Cta_tile>>
struct Smem_tile_mma_transposed : public Base {
  enum { BYTES_PER_LDS = 16 };
  enum { BYTES_PER_ROW = Base::BYTES_PER_ROW };
  enum { BYTES_PER_ELT = Base::BYTES_PER_ELT };
  enum { WARPS_M = Base::WARPS_M };
  enum { WARPS_N = Base::WARPS_N };
  static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
  using Fragment = typename Base::Fragment;
  inline __device__ Smem_tile_mma_transposed(char *smem, int tidx)
      : Base(smem, tidx) {
    static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
    int read_row, read_col;
    read_row = (tidx & 0x0f);
    read_col = (tidx & 0xe0) / 16 + (tidx & 0x1c) / 16;

    read_col ^= (read_row & 0x07);
    read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
  }

  template <int M, int N>
  inline __device__ void load(Fragment (&frag)[M][N]) {
    static_assert(Base::COLS == Cta_tile::N);
    for (int mi = 0; mi < M; mi++) {
      for (int ni = 0; ni < N; ni++) {
        size_t offset = read_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW +
                        ni * WARPS_N * 16 * BYTES_PER_ELT;
        uint4 dst;
        fmha::ldsmt(dst, this->smem_ + offset);
        frag[mi][ni].reg(0) = dst.x;
        frag[mi][ni].reg(1) = dst.z;  // Fragment A regs col major!
        frag[mi][ni].reg(2) = dst.y;
        frag[mi][ni].reg(3) = dst.w;
      }
    }
  }

  uint32_t read_offset_;
};

template <typename Cta_tile, typename Base = Smem_tile_mma<Cta_tile>>
struct Smem_tile_mma_epilogue : public Base {
  enum { BYTES_PER_LDS = 16 };
  enum { BYTES_PER_ROW = Base::BYTES_PER_ROW };
  enum { BYTES_PER_ELT = Base::BYTES_PER_ELT };
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDS };
  static_assert(THREADS_PER_ROW * BYTES_PER_LDS == BYTES_PER_ROW);
  enum { ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
  enum { NUM_LDS = Cta_tile::M / ROWS_PER_LDS };
  static_assert(NUM_LDS * ROWS_PER_LDS == Cta_tile::M);
  enum { WARPS_M = Base::WARPS_M };
  enum { WARPS_N = Base::WARPS_N };
  static_assert((WARPS_M == 4 || WARPS_N == 8) || WARPS_N == 1);
  using Fragment = typename Base::Fragment;

  inline __device__ Smem_tile_mma_epilogue(char *smem, int tidx)
      : Base(smem, tidx) {
    const int read_row = tidx / THREADS_PER_ROW;
    int read_col = tidx % THREADS_PER_ROW;
    read_col ^= (read_row & 0x07);
    read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
  }

  inline __device__ void load(uint4 (&data)[NUM_LDS]) {
    for (int ii = 0; ii < NUM_LDS; ii++) {
      size_t offset = read_offset_ + ii * ROWS_PER_LDS * BYTES_PER_ROW;
      fmha::lds(data[ii], this->smem_ + offset);
    }
  }

  template <int M, int N>
  inline __device__ void store(const uint4 (&regs)[M][N]) {
    for (int mi = 0; mi < M; mi++) {
      for (int ni = 0; ni < N; ni++) {
        size_t offset = (this->write_offset_ ^ (ni * 32)) +
                        mi * WARPS_M * 16 * BYTES_PER_ROW;
        fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
        fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
        offset ^= 4 * Base::BYTES_PER_STS;
        fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
        fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
      }
    }
  }

  uint32_t read_offset_;
};

}  // namespace fmha
