/*
 *  Copyright 2008-20120 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/shuffle.h>

#include <cstdint>

namespace thrust {
template <typename Iterator>
using iterator_value_t = typename iterator_value<Iterator>::type;

namespace system {
namespace detail {
namespace generic {

// An implementation of a Feistel cipher for operating on 64 bit keys
class feistel_bijection {
  struct round_state {
    std::uint32_t left;
    std::uint32_t right;
  };

 public:
  template <class URBG>
  __host__ __device__ feistel_bijection(std::uint64_t m, URBG&& g) {
    std::uint64_t total_bits = get_cipher_bits(m);
    // Half bits rounded down
    left_side_bits = total_bits / 2;
    left_side_mask = (1ull << left_side_bits) - 1;
    // Half the bits rounded up
    right_side_bits = total_bits - left_side_bits;
    right_side_mask = (1ull << right_side_bits) - 1;

    for (std::uint64_t i = 0; i < num_rounds; i++) {
      key[i] = g();
    }
  }

  __host__ __device__ std::uint64_t nearest_power_of_two() const {
    return 1ull << (left_side_bits + right_side_bits);
  }
  __host__ __device__ std::uint64_t operator()(const std::uint64_t val) const {
    // Extract the right and left sides of the input
    auto left = static_cast<std::uint32_t>(val >> right_side_bits);
    auto right = static_cast<std::uint32_t>(val & right_side_mask);
    round_state state = {left, right};

    for (std::uint64_t i = 0; i < num_rounds; i++) {
      state = do_round(state, i);
    }

    // Check we have the correct number of bits on each side
    assert((state.left >> left_side_bits) == 0);
    assert((state.right >> right_side_bits) == 0);

    // Combine the left and right sides together to get result
    return state.left << right_side_bits | state.right;
  }

 private:
  // Find the nearest power of two
  __host__ __device__ std::uint64_t get_cipher_bits(std::uint64_t m) {
    if (m == 0) return 0;
    std::uint64_t i = 0;
    m--;
    while (m != 0) {
      i++;
      m >>= 1;
    }
    return i;
  }

  // Equivalent to boost::hash_combine
  __host__ __device__
  std::size_t hash_combine(std::uint64_t lhs, std::uint64_t rhs) const {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }

  // Round function, a 'pseudorandom function' who's output is indistinguishable
  // from random for each key value input. This is not cryptographically secure
  // but sufficient for generating permutations. 
  __host__ __device__ std::uint32_t round_function(std::uint64_t value,
                                              const std::uint64_t key_) const {
    std::uint64_t hash0 = thrust::random::taus88(static_cast<std::uint32_t>(value))();
    std::uint64_t hash1 = thrust::random::ranlux48(value)();
    return static_cast<std::uint32_t>(
      hash_combine(hash_combine(hash0, key_), hash1) & left_side_mask);
  }

  __host__ __device__ round_state do_round(const round_state state,
                                           const std::uint64_t round) const {
    const std::uint32_t new_left = state.right & left_side_mask;
    const std::uint32_t round_function_res =
        state.left ^ round_function(state.right, key[round]);
    if (right_side_bits != left_side_bits) {
      // Upper bit of the old right becomes lower bit of new right if we have
      // odd length feistel
      const std::uint32_t new_right =
          (round_function_res << 1ull) | state.right >> left_side_bits;
      return {new_left, new_right};
    }
    return {new_left, round_function_res};
  }

  static constexpr std::uint64_t num_rounds = 16;
  std::uint64_t right_side_bits;
  std::uint64_t left_side_bits;
  std::uint64_t right_side_mask;
  std::uint64_t left_side_mask;
  std::uint64_t key[num_rounds];
};

struct key_flag_tuple {
  std::uint64_t key;
  std::uint64_t flag;
};

// scan only flags
struct key_flag_scan_op {
  __host__ __device__ key_flag_tuple operator()(const key_flag_tuple& a,
                                                const key_flag_tuple& b) {
    return {b.key, a.flag + b.flag};
  }
};

struct construct_key_flag_op {
  std::uint64_t m;
  feistel_bijection bijection;
  __host__ __device__ construct_key_flag_op(std::uint64_t m,
                                            feistel_bijection bijection)
      : m(m), bijection(bijection) {}
  __host__ __device__ key_flag_tuple operator()(std::uint64_t idx) {
    auto gather_key = bijection(idx);
    return key_flag_tuple{gather_key, (gather_key < m) ? 1ull : 0ull};
  }
};

template <typename InputIterT, typename OutputIterT>
struct write_output_op {
  std::uint64_t m;
  InputIterT in;
  OutputIterT out;
  // flag contains inclusive scan of valid keys
  // perform gather using valid keys
  __thrust_exec_check_disable__
  __host__ __device__ std::size_t operator()(key_flag_tuple x) {
    if (x.key < m) {
      // -1 because inclusive scan
      out[x.flag - 1] = in[x.key];
    }
    return 0;  // Discarded
  }
};

template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, URBG&& g) {
  using InputType = typename thrust::iterator_value_t<RandomIterator>;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, ExecutionPolicy> temp(exec, first,
                                                                   last);
  thrust::shuffle_copy(exec, temp.begin(), temp.end(), first, g);
}

template <typename ExecutionPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, OutputIterator result, URBG&& g) {
  // m is the length of the input
  // we have an available bijection of length n via a feistel cipher
  std::size_t m = last - first;
  feistel_bijection bijection(m, g);
  std::uint64_t n = bijection.nearest_power_of_two();

  // perform stream compaction over length n bijection to get length m
  // pseudorandom bijection over the original input
  thrust::counting_iterator<std::uint64_t> indices(0);
  thrust::transform_iterator<construct_key_flag_op, decltype(indices),
                             key_flag_tuple>
      key_flag_it(indices, construct_key_flag_op(m, bijection));
  write_output_op<RandomIterator, decltype(result)> write_functor{m, first,
                                                                  result};
  auto gather_output_it = thrust::make_transform_output_iterator(
      thrust::discard_iterator<std::size_t>(), write_functor);
  // the feistel_bijection outputs a stream of permuted indices in range [0,n)
  // flag each value < m and compact it, so we have a set of permuted indices in
  // range [0,m) each thread gathers an input element according to its
  // pseudorandom permuted index
  thrust::inclusive_scan(exec, key_flag_it, key_flag_it + n, gather_output_it,
                         key_flag_scan_op());
}

}  // namespace generic
}  // namespace detail
}  // namespace system
}  // namespace thrust
