#include "moe_kernel_impl.h"
namespace phi{
  // ===== CUB Sorting things =====
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

CubKeyValueSorter::CubKeyValueSorter(cudaStream_t stream)
    : num_experts_(0), num_bits_(sizeof(int) * 8), stream_(stream) {}

CubKeyValueSorter::CubKeyValueSorter(const int num_experts)
    : num_experts_(num_experts),
      num_bits_(static_cast<int>(log2(num_experts)) + 1) {}

void CubKeyValueSorter::update_num_experts(const int num_experts) {
  num_experts_ = num_experts;
  num_bits_ = static_cast<int>(log2(num_experts)) +
              3;  // 额外增加 3 位用于标记 topk的位置
}

size_t CubKeyValueSorter::getWorkspaceSize(const size_t num_key_value_pairs,
                                           bool descending) {
  num_key_value_pairs_ = num_key_value_pairs;
  size_t required_storage = 0;
  int* null_int = nullptr;
  if (descending) {
    cub::DeviceRadixSort::SortPairsDescending(NULL,
                                              required_storage,
                                              null_int,
                                              null_int,
                                              null_int,
                                              null_int,
                                              num_key_value_pairs,
                                              0,
                                              32,
                                              stream_);
  } else {
    cub::DeviceRadixSort::SortPairs(NULL,
                                    required_storage,
                                    null_int,
                                    null_int,
                                    null_int,
                                    null_int,
                                    num_key_value_pairs,
                                    0,
                                    num_bits_,
                                    stream_);
  }
  return required_storage;
}
}