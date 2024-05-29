/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>

namespace paddle {
namespace framework {

template<
    typename KeyT,
    int BLOCK_SIZE,
    bool GREATER = true,
    int RADIX_BITS = 8>
class BlockRadixTopKGlobalMemory {
  static_assert(cub::PowerOfTwo<RADIX_BITS>::VALUE && (RADIX_BITS <= (sizeof(KeyT) * 8)),
                "RADIX_BITS should be power of 2, and <= (sizeof(KeyT) * 8)");
  static_assert(cub::PowerOfTwo<BLOCK_SIZE>::VALUE, "BLOCK_SIZE should be power of 2");
  using KeyTraits = cub::Traits<KeyT>;
  using UnsignedBits = typename KeyTraits::UnsignedBits;
  using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;
  static constexpr int RADIX_SIZE = (1 << RADIX_BITS);
  static constexpr int SCAN_ITEMS_PER_THREAD = (RADIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using BinBlockLoad = cub::BlockLoad<int, BLOCK_SIZE, SCAN_ITEMS_PER_THREAD>;
  using BinBlockStore = cub::BlockStore<int, BLOCK_SIZE, SCAN_ITEMS_PER_THREAD>;
  struct _TempStorage {
    typename BlockScanT::TempStorage scan_storage;
    union {
      typename BinBlockLoad::TempStorage load_storage;
      typename BinBlockStore::TempStorage store_storage;
    } load_store;
    union {
      int shared_bins[RADIX_SIZE];
    };
    int share_target_k;
    int share_bucket_id;
  };

 public:
  struct TempStorage : cub::Uninitialized<_TempStorage> {
  };
  __device__ __forceinline__ BlockRadixTopKGlobalMemory(TempStorage &temp_storage)
      : temp_storage_{temp_storage.Alias()}, tid_(threadIdx.x){};
  __device__ __forceinline__ void radixTopKGetThreshold(const KeyT *data, int k, int size, KeyT &topK, bool &topk_is_unique) {
    assert(k < size && k > 0);
    int target_k = k;
    UnsignedBits key_pattern = 0;
    int digit_pos = sizeof(KeyT) * 8 - RADIX_BITS;
    for (; digit_pos >= 0; digit_pos -= RADIX_BITS) {
      UpdateSharedBins(data, size, digit_pos, key_pattern);
      InclusiveScanBins();
      UpdateTopK(digit_pos, target_k, key_pattern);
      if (target_k == 0) break;
    }
    if (target_k == 0) {
      key_pattern -= 1;
      topk_is_unique = true;
    } else {
      topk_is_unique = false;
    }
    if (GREATER) key_pattern = ~key_pattern;
    UnsignedBits topK_unsigned = KeyTraits::TwiddleOut(key_pattern);
    topK = reinterpret_cast<KeyT &>(topK_unsigned);
  }

 private:
  __device__ __forceinline__ void UpdateSharedBins(const KeyT *key, int size, int digit_pos, UnsignedBits key_pattern) {
    for (int id = tid_; id < RADIX_SIZE; id += BLOCK_SIZE) {
      temp_storage_.shared_bins[id] = 0;
    }
    cub::CTA_SYNC();
    UnsignedBits key_mask = ((UnsignedBits)(-1)) << ((UnsignedBits)(digit_pos + RADIX_BITS));
#pragma unroll
    for (int idx = tid_; idx < size; idx += BLOCK_SIZE) {
      KeyT key_data = key[idx];
      UnsignedBits twiddled_data = KeyTraits::TwiddleIn(reinterpret_cast<UnsignedBits &>(key_data));
      if (GREATER) twiddled_data = ~twiddled_data;
      UnsignedBits digit_in_radix = cub::BFE<UnsignedBits>(twiddled_data, digit_pos, RADIX_BITS);
      if ((twiddled_data & key_mask) == (key_pattern & key_mask)) {
        atomicAdd(&temp_storage_.shared_bins[digit_in_radix], 1);
      }
    }
    cub::CTA_SYNC();
  }
  __device__ __forceinline__ void InclusiveScanBins() {
    int items[SCAN_ITEMS_PER_THREAD];
    BinBlockLoad(temp_storage_.load_store.load_storage).Load(temp_storage_.shared_bins, items, RADIX_SIZE, 0);
    cub::CTA_SYNC();
    BlockScanT(temp_storage_.scan_storage).InclusiveSum(items, items);
    cub::CTA_SYNC();
    BinBlockStore(temp_storage_.load_store.store_storage).Store(temp_storage_.shared_bins, items, RADIX_SIZE);
    cub::CTA_SYNC();
  }
  __device__ __forceinline__ void UpdateTopK(int digit_pos,
                                             int &target_k,
                                             UnsignedBits &target_pattern) {
    for (int idx = tid_; (idx < RADIX_SIZE); idx += BLOCK_SIZE) {
      int prev_count = (idx == 0) ? 0 : temp_storage_.shared_bins[idx - 1];
      int cur_count = temp_storage_.shared_bins[idx];
      if (prev_count <= target_k && cur_count > target_k) {
        temp_storage_.share_target_k = target_k - prev_count;
        temp_storage_.share_bucket_id = idx;
      }
    }
    cub::CTA_SYNC();
    target_k = temp_storage_.share_target_k;
    int target_bucket_id = temp_storage_.share_bucket_id;
    UnsignedBits key_segment = ((UnsignedBits) target_bucket_id) << ((UnsignedBits) digit_pos);
    target_pattern |= key_segment;
  }
  _TempStorage &temp_storage_;
  int tid_;
};

template<
    typename KeyT,
    int BLOCK_SIZE,
    int ITEMS_PER_THREAD,
    bool GREATER = true,
    typename ValueT = cub::NullType,
    int RADIX_BITS = 8>
class BlockRadixTopKRegister {
  static_assert(cub::PowerOfTwo<RADIX_BITS>::VALUE && (RADIX_BITS <= (sizeof(KeyT) * 8)),
                "RADIX_BITS should be power of 2, and <= (sizeof(KeyT) * 8)");
  static_assert(cub::PowerOfTwo<BLOCK_SIZE>::VALUE, "BLOCK_SIZE should be power of 2");
  using KeyTraits = cub::Traits<KeyT>;
  using UnsignedBits = typename KeyTraits::UnsignedBits;
  using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;
  static constexpr int RADIX_SIZE = (1 << RADIX_BITS);
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;
  static constexpr int SCAN_ITEMS_PER_THREAD = (RADIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using BinBlockLoad = cub::BlockLoad<int, BLOCK_SIZE, SCAN_ITEMS_PER_THREAD>;
  using BinBlockStore = cub::BlockStore<int, BLOCK_SIZE, SCAN_ITEMS_PER_THREAD>;
  using BlockExchangeKey = cub::BlockExchange<KeyT, BLOCK_SIZE, ITEMS_PER_THREAD>;
  using BlockExchangeValue = cub::BlockExchange<ValueT, BLOCK_SIZE, ITEMS_PER_THREAD>;

  using _ExchangeKeyTempStorage = typename BlockExchangeKey::TempStorage;
  using _ExchangeValueTempStorage = typename BlockExchangeValue::TempStorage;
  typedef union ExchangeKeyTempStorageType {
    _ExchangeKeyTempStorage key_storage;
  } ExchKeyTempStorageType;
  typedef union ExchangeKeyValueTempStorageType {
    _ExchangeKeyTempStorage key_storage;
    _ExchangeValueTempStorage value_storage;
  } ExchKeyValueTempStorageType;
  using _ExchangeType = typename std::conditional<KEYS_ONLY, ExchKeyTempStorageType, ExchKeyValueTempStorageType>::type;

  struct _TempStorage {
    typename BlockScanT::TempStorage scan_storage;
    union {
      typename BinBlockLoad::TempStorage load_storage;
      typename BinBlockStore::TempStorage store_storage;
    } load_store;
    union {
      int shared_bins[RADIX_SIZE];
      _ExchangeType exchange_storage;
    };
    int share_target_k;
    int share_bucket_id;
    int share_prev_count;
  };

 public:
  struct TempStorage : cub::Uninitialized<_TempStorage> {
  };
  __device__ __forceinline__ BlockRadixTopKRegister(TempStorage &temp_storage)
      : temp_storage_{temp_storage.Alias()}, tid_(threadIdx.x){};
  __device__ __forceinline__ void radixTopKToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                     const int k, const int valid_count) {
    TopKGenRank(keys, k, valid_count);
    int is_valid[ITEMS_PER_THREAD];
    GenValidArray(is_valid, k);
    BlockExchangeKey{temp_storage_.exchange_storage.key_storage}.ScatterToStripedFlagged(keys, keys, ranks_, is_valid);
    cub::CTA_SYNC();
  }
  __device__ __forceinline__ void radixTopKToStriped(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD],
                                                     const int k, const int valid_count) {
    TopKGenRank(keys, k, valid_count);
    int is_valid[ITEMS_PER_THREAD];
    GenValidArray(is_valid, k);
    BlockExchangeKey{temp_storage_.exchange_storage.key_storage}.ScatterToStripedFlagged(keys, keys, ranks_, is_valid);
    cub::CTA_SYNC();
    BlockExchangeValue{temp_storage_.exchange_storage.value_storage}.ScatterToStripedFlagged(values, values, ranks_, is_valid);
    cub::CTA_SYNC();
  }

 private:
  __device__ __forceinline__ void TopKGenRank(KeyT (&keys)[ITEMS_PER_THREAD], const int k, const int valid_count) {
    assert(k <= BLOCK_SIZE * ITEMS_PER_THREAD);
    assert(k <= valid_count);
    if (k == valid_count) return;
    UnsignedBits(&unsigned_keys)[ITEMS_PER_THREAD] = reinterpret_cast<UnsignedBits(&)[ITEMS_PER_THREAD]>(keys);
    search_mask_ = 0;
    top_k_mask_ = 0;

#pragma unroll
    for (unsigned int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      int idx = KEY * BLOCK_SIZE + tid_;
      unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
      if (GREATER) unsigned_keys[KEY] = ~unsigned_keys[KEY];
      if (idx < valid_count) search_mask_ |= (1U << KEY);
    }

    int target_k = k;
    int prefix_k = 0;

    for (int digit_pos = sizeof(KeyT) * 8 - RADIX_BITS; digit_pos >= 0; digit_pos -= RADIX_BITS) {
      UpdateSharedBins(unsigned_keys, digit_pos, prefix_k);
      InclusiveScanBins();
      UpdateTopK(unsigned_keys, digit_pos, target_k, prefix_k, digit_pos == 0);
      if (target_k == 0) break;
    }

#pragma unroll
    for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      if (GREATER) unsigned_keys[KEY] = ~unsigned_keys[KEY];
      unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
    }
  }
  __device__ __forceinline__ void GenValidArray(int (&is_valid)[ITEMS_PER_THREAD], int k) {
#pragma unroll
    for (unsigned int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      if ((top_k_mask_ & (1U << KEY)) && ranks_[KEY] < k) {
        is_valid[KEY] = 1;
      } else {
        is_valid[KEY] = 0;
      }
    }
  }
  __device__ __forceinline__ void UpdateSharedBins(UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD],
                                                   int digit_pos, int prefix_k) {
    for (int id = tid_; id < RADIX_SIZE; id += BLOCK_SIZE) {
      temp_storage_.shared_bins[id] = 0;
    }
    cub::CTA_SYNC();
//#define USE_MATCH
#ifdef USE_MATCH
    int lane_mask = cub::LaneMaskLt();
#pragma unroll
    for (unsigned int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      bool is_search = search_mask_ & (1U << KEY);
      int bucket_idx = -1;
      if (is_search) {
        UnsignedBits digit_in_radix = cub::BFE<UnsignedBits>(unsigned_keys[KEY], digit_pos, RADIX_BITS);
        bucket_idx = (int) digit_in_radix;
      }
      int warp_match_mask = __match_any_sync(0xffffffff, bucket_idx);
      int same_count = __popc(warp_match_mask);
      int idx_in_same_bucket = __popc(warp_match_mask & lane_mask);
      int same_bucket_root_lane = __ffs(warp_match_mask) - 1;
      int same_bucket_start_idx;
      if (idx_in_same_bucket == 0 && is_search) {
        same_bucket_start_idx = atomicAdd(&temp_storage_.shared_bins[bucket_idx], same_count);
      }
      same_bucket_start_idx = __shfl_sync(0xffffffff, same_bucket_start_idx, same_bucket_root_lane, 32);
      if (is_search) {
        ranks_[KEY] = same_bucket_start_idx + idx_in_same_bucket + prefix_k;
      }
    }
#else
#pragma unroll
    for (unsigned int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      bool is_search = search_mask_ & (1U << KEY);
      int bucket_idx = -1;
      if (is_search) {
        UnsignedBits digit_in_radix = cub::BFE<UnsignedBits>(unsigned_keys[KEY], digit_pos, RADIX_BITS);
        bucket_idx = (int) digit_in_radix;
        ranks_[KEY] = atomicAdd(&temp_storage_.shared_bins[bucket_idx], 1) + prefix_k;
      }
    }
#endif
    cub::CTA_SYNC();
  }
  __device__ __forceinline__ void InclusiveScanBins() {
    int items[SCAN_ITEMS_PER_THREAD];
    BinBlockLoad(temp_storage_.load_store.load_storage).Load(temp_storage_.shared_bins, items, RADIX_SIZE, 0);
    cub::CTA_SYNC();
    BlockScanT(temp_storage_.scan_storage).InclusiveSum(items, items);
    cub::CTA_SYNC();
    BinBlockStore(temp_storage_.load_store.store_storage).Store(temp_storage_.shared_bins, items, RADIX_SIZE);
    cub::CTA_SYNC();
  }
  __device__ __forceinline__ void UpdateTopK(UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD],
                                             int digit_pos,
                                             int &target_k,
                                             int &prefix_k,
                                             bool mark_equal) {
    for (int idx = tid_; (idx < RADIX_SIZE); idx += BLOCK_SIZE) {
      int prev_count = (idx == 0) ? 0 : temp_storage_.shared_bins[idx - 1];
      int cur_count = temp_storage_.shared_bins[idx];
      if (prev_count <= target_k && cur_count > target_k) {
        temp_storage_.share_target_k = target_k - prev_count;
        temp_storage_.share_bucket_id = idx;
        temp_storage_.share_prev_count = prev_count;
      }
    }
    cub::CTA_SYNC();
    target_k = temp_storage_.share_target_k;
    prefix_k += temp_storage_.share_prev_count;
    int target_bucket_id = temp_storage_.share_bucket_id;
#pragma unroll
    for (unsigned int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++) {
      if (search_mask_ & (1U << KEY)) {
        UnsignedBits digit_in_radix = cub::BFE<UnsignedBits>(unsigned_keys[KEY], digit_pos, RADIX_BITS);
        if (digit_in_radix < target_bucket_id) {
          top_k_mask_ |= (1U << KEY);
          search_mask_ &= ~(1U << KEY);
        } else if (digit_in_radix > target_bucket_id) {
          search_mask_ &= ~(1U << KEY);
        } else {
          if (mark_equal) top_k_mask_ |= (1U << KEY);
        }
        if (digit_in_radix <= target_bucket_id) {
          int prev_count = (digit_in_radix == 0) ? 0 : temp_storage_.shared_bins[digit_in_radix - 1];
          ranks_[KEY] += prev_count;
        }
      }
    }
    cub::CTA_SYNC();
  }

  _TempStorage &temp_storage_;
  int tid_;
  int ranks_[ITEMS_PER_THREAD];
  unsigned int search_mask_;
  unsigned int top_k_mask_;
};

};  // namespace framework
};  // namespace paddle
