// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Unit test of the gin_overlap sum kernels (gin_overlap/kernels/sum.h).
// Every result is compared bit for bit with a CPU reference that adds the
// sources in the same ring order and with the same float roundings.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/sum.h"

namespace paddle {
namespace distributed {
namespace {

#define CHECK_CUDA(call) CheckCuda((call), #call, __FILE__, __LINE__)

void CheckCuda(cudaError_t error,
               const char* call,
               const char* file,
               int line) {
  if (error != cudaSuccess) {
    std::fprintf(stderr,
                 "%s:%d: %s failed: %s\n",
                 file,
                 line,
                 call,
                 cudaGetErrorString(error));
    std::abort();
  }
}

// Floats on the device, starting offset_floats floats after a cudaMalloc
// address (cudaMalloc addresses are at least 256-byte aligned, so offset 0 is
// 16-byte aligned and offset 1 is not).
class DeviceFloats {
 public:
  DeviceFloats(const std::vector<float>& host_values, size_t offset_floats)
      : size_(host_values.size()), offset_floats_(offset_floats) {
    CHECK_CUDA(
        cudaMalloc(&allocation_, (offset_floats + size_ + 1) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data(),
                          host_values.data(),
                          size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  ~DeviceFloats() { CHECK_CUDA(cudaFree(allocation_)); }
  DeviceFloats(const DeviceFloats&) = delete;
  DeviceFloats& operator=(const DeviceFloats&) = delete;

  float* data() { return allocation_ + offset_floats_; }

  std::vector<float> CopyToHost() const {
    std::vector<float> host_values(size_);
    CHECK_CUDA(cudaMemcpy(host_values.data(),
                          allocation_ + offset_floats_,
                          size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host_values;
  }

 private:
  float* allocation_ = nullptr;
  size_t size_;
  size_t offset_floats_;
};

// Random floats with exponents from -24 to 24 and both signs, so that adding
// them in a different order usually changes the low bits of the sum.
std::vector<float> RandomFloats(size_t count, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
  std::uniform_int_distribution<int> exponent(-24, 24);
  std::bernoulli_distribution negative(0.5);
  std::vector<float> values(count);
  for (float& value : values) {
    value = std::ldexp(mantissa(generator), exponent(generator));
    if (negative(generator)) value = -value;
  }
  return values;
}

// The kernel's sum of one element: ring order from self_index + 1, starting at
// 0. With use_scale, the product and the sum are each rounded to float (the
// volatile temporaries keep the host compiler from fusing them into an FMA).
template <typename ValueOfSource>
float ReferenceRingSum(int num_sources,
                       int self_index,
                       bool use_scale,
                       float scale,
                       ValueOfSource value_of_source) {
  float sum = 0.0f;
  for (int ring_step = 0; ring_step < num_sources; ++ring_step) {
    const int source = (self_index + 1 + ring_step) % num_sources;
    if (use_scale) {
      volatile float product = value_of_source(source) * scale;
      volatile float rounded_sum = sum + product;
      sum = rounded_sum;
    } else {
      sum += value_of_source(source);
    }
  }
  return sum;
}

void ExpectSameBits(const std::vector<float>& expected,
                    const std::vector<float>& actual,
                    const std::string& context) {
  ASSERT_EQ(expected.size(), actual.size()) << context;
  if (std::memcmp(expected.data(),
                  actual.data(),
                  expected.size() * sizeof(float)) == 0) {
    return;
  }
  for (size_t index = 0; index < expected.size(); ++index) {
    uint32_t expected_bits;
    uint32_t actual_bits;
    std::memcpy(&expected_bits, &expected[index], sizeof(float));
    std::memcpy(&actual_bits, &actual[index], sizeof(float));
    if (expected_bits != actual_bits) {
      ADD_FAILURE() << context << ": first mismatch at element " << index
                    << ", expected " << expected[index] << " (0x" << std::hex
                    << expected_bits << "), got " << actual[index] << " (0x"
                    << actual_bits << ")";
      return;
    }
  }
}

struct FinalSumCase {
  size_t numel;
  int num_sources;
  int self_index;
  float scale;
  size_t shard_offset = 0;  // in floats; 1 makes shard unaligned
  size_t recv_offset = 0;   // in floats; 1 makes recv unaligned
  int num_blocks = 4;
  int num_threads = 256;
};

// Runs LaunchFinalSum on random inputs: the shard must equal the reference bit
// for bit and recv must be unchanged.
void CheckFinalSum(const FinalSumCase& test_case) {
  const size_t numel = test_case.numel;
  const int num_sources = test_case.num_sources;
  const int self_index = test_case.self_index;
  std::ostringstream context;
  context << "LaunchFinalSum numel=" << numel << " num_sources=" << num_sources
          << " self_index=" << self_index << " scale=" << test_case.scale
          << " shard_offset=" << test_case.shard_offset
          << " recv_offset=" << test_case.recv_offset
          << " num_blocks=" << test_case.num_blocks
          << " num_threads=" << test_case.num_threads;

  const std::vector<float> shard = RandomFloats(numel, 1);
  // Row self_index of recv is random too: a kernel that read it would fail.
  const std::vector<float> recv = RandomFloats(num_sources * numel, 2);
  std::vector<float> expected(numel);
  for (size_t element = 0; element < numel; ++element) {
    expected[element] = ReferenceRingSum(
        num_sources,
        self_index,
        /*use_scale=*/true,
        test_case.scale,
        [&](int source) {
          return source == self_index ? shard[element]
                                      : recv[source * numel + element];
        });
  }

  DeviceFloats device_shard(shard, test_case.shard_offset);
  DeviceFloats device_recv(recv, test_case.recv_offset);
  ASSERT_EQ(cudaSuccess,
            LaunchFinalSum(device_shard.data(),
                           device_recv.data(),
                           numel,
                           num_sources,
                           self_index,
                           test_case.scale,
                           test_case.num_blocks,
                           /*stream=*/nullptr,
                           test_case.num_threads))
      << context.str();
  CHECK_CUDA(cudaDeviceSynchronize());
  ExpectSameBits(expected, device_shard.CopyToHost(), context.str());
  ExpectSameBits(recv, device_recv.CopyToHost(), context.str() + " (recv)");
}

struct IntraNodeSumCase {
  size_t numel;
  int ranks_per_node;
  int num_nodes;
  int rank_in_node;
  size_t grad_offset = 0;  // in floats; 1 makes grad unaligned
  size_t recv_offset = 0;  // in floats; 1 makes intra_recv unaligned
  int num_blocks = 4;
  int num_threads = 512;
};

// Runs LaunchIntraNodeSum on random inputs: block (node, rank_in_node) of each
// node must equal the reference bit for bit, and the other blocks of grad and
// all of intra_recv must be unchanged.
void CheckIntraNodeSum(const IntraNodeSumCase& test_case) {
  const size_t numel = test_case.numel;
  const int ranks_per_node = test_case.ranks_per_node;
  const int num_nodes = test_case.num_nodes;
  const int rank_in_node = test_case.rank_in_node;
  std::ostringstream context;
  context << "LaunchIntraNodeSum numel=" << numel
          << " ranks_per_node=" << ranks_per_node << " num_nodes=" << num_nodes
          << " rank_in_node=" << rank_in_node
          << " grad_offset=" << test_case.grad_offset
          << " recv_offset=" << test_case.recv_offset
          << " num_blocks=" << test_case.num_blocks
          << " num_threads=" << test_case.num_threads;

  // grad: [num_nodes][ranks_per_node][numel]
  // intra_recv: [ranks_per_node - 1][num_nodes][numel], no row for rank_in_node
  const std::vector<float> grad =
      RandomFloats(num_nodes * ranks_per_node * numel, 3);
  const std::vector<float> intra_recv =
      RandomFloats((ranks_per_node - 1) * num_nodes * numel, 4);
  std::vector<float> expected = grad;
  for (int node = 0; node < num_nodes; ++node) {
    for (size_t element = 0; element < numel; ++element) {
      const size_t self_position =
          (node * ranks_per_node + rank_in_node) * numel + element;
      expected[self_position] = ReferenceRingSum(
          ranks_per_node,
          rank_in_node,
          /*use_scale=*/false,
          1.0f,
          [&](int source) {
            if (source == rank_in_node) return grad[self_position];
            const int row = source - (source > rank_in_node);
            return intra_recv[(row * num_nodes + node) * numel + element];
          });
    }
  }

  DeviceFloats device_grad(grad, test_case.grad_offset);
  DeviceFloats device_recv(intra_recv, test_case.recv_offset);
  ASSERT_EQ(cudaSuccess,
            LaunchIntraNodeSum(device_grad.data(),
                               device_recv.data(),
                               numel,
                               ranks_per_node,
                               num_nodes,
                               rank_in_node,
                               test_case.num_blocks,
                               /*stream=*/nullptr,
                               test_case.num_threads))
      << context.str();
  CHECK_CUDA(cudaDeviceSynchronize());
  ExpectSameBits(expected, device_grad.CopyToHost(), context.str());
  ExpectSameBits(
      intra_recv, device_recv.CopyToHost(), context.str() + " (intra_recv)");
}

// Indices of this rank among num_sources sources to test: all of them up to
// 8, otherwise the first, a middle one, the last and the two around the
// batch boundary of the kernel (8 sources per batch).
std::vector<int> SelfIndices(int num_sources) {
  if (num_sources <= 8) {
    std::vector<int> indices(num_sources);
    for (int index = 0; index < num_sources; ++index) indices[index] = index;
    return indices;
  }
  std::vector<int> indices = {0, 7, 8, num_sources / 2, num_sources - 1};
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

}  // namespace

// The bitwise checks below only catch a wrong summation order if the inputs
// are order sensitive: starting the ring at a different source must change
// some result bits.
TEST(GinOverlapSumTest, RandomInputsAreOrderSensitive) {
  const int num_sources = 8;
  const size_t numel = 1000;
  const std::vector<float> values = RandomFloats(num_sources * numel, 2);
  int num_different = 0;
  for (size_t element = 0; element < numel; ++element) {
    auto value_of_source = [&](int source) {
      return values[source * numel + element];
    };
    const float from_source_1 = ReferenceRingSum(
        num_sources, 0, /*use_scale=*/false, 1.0f, value_of_source);
    const float from_source_2 = ReferenceRingSum(
        num_sources, 1, /*use_scale=*/false, 1.0f, value_of_source);
    num_different +=
        std::memcmp(&from_source_1, &from_source_2, sizeof(float)) != 0;
  }
  EXPECT_GT(num_different, 0);
}

// num_sources above 8 spans several batches of the kernel, with a partial
// last batch for 9 and 17 (16 is the largest expert group / node count
// tested).
TEST(GinOverlapSumTest, FinalSumMatchesReference) {
  for (int num_sources : {1, 2, 3, 4, 8, 9, 16, 17}) {
    for (int self_index : SelfIndices(num_sources)) {
      for (float scale : {1.0f, 1.0f / num_sources, 0.3f}) {
        CheckFinalSum({/*numel=*/1000, num_sources, self_index, scale});
      }
    }
  }
}

TEST(GinOverlapSumTest, IntraNodeSumMatchesReference) {
  // 16 is the largest ranks_per_node of the scheduler (kMaxRanksPerNode).
  for (int ranks_per_node : {1, 2, 8, 9, 16}) {
    for (int num_nodes : {1, 2, 4}) {
      for (int rank_in_node : SelfIndices(ranks_per_node)) {
        CheckIntraNodeSum(
            {/*numel=*/1000, ranks_per_node, num_nodes, rank_in_node});
      }
    }
  }
}

// numel 4 and 1000 with aligned pointers take the float4 path; numel 1, 3 and
// 4097, or a pointer offset by one float, take the scalar path. 3 blocks of 64
// threads do not divide any of these sizes. 8 sources fill one batch of the
// float4 path; 9 and 17 add a partial batch, 16 a second full one.
TEST(GinOverlapSumTest, Float4AndScalarPathsMatchReference) {
  struct PointerOffsets {
    size_t self_offset;
    size_t recv_offset;
  };
  for (int num_sources : {8, 9, 16, 17}) {
    for (size_t numel : {1, 3, 4, 1000, 4097}) {
      for (PointerOffsets offsets :
           {PointerOffsets{0, 0}, PointerOffsets{1, 0}, PointerOffsets{0, 1}}) {
        CheckFinalSum({numel,
                       num_sources,
                       /*self_index=*/3,
                       /*scale=*/0.3f,
                       offsets.self_offset,
                       offsets.recv_offset,
                       /*num_blocks=*/3,
                       /*num_threads=*/64});
        CheckIntraNodeSum({numel,
                           /*ranks_per_node=*/num_sources,
                           /*num_nodes=*/2,
                           /*rank_in_node=*/5,
                           offsets.self_offset,
                           offsets.recv_offset,
                           /*num_blocks=*/3,
                           /*num_threads=*/64});
      }
    }
  }
}

// Every grid gives the reference bits, so all grids give the same bits.
TEST(GinOverlapSumTest, BlockAndThreadCountsDoNotChangeBits) {
  for (int num_sources : {8, 17}) {      // one batch, a partial third batch
    for (size_t numel : {1000, 4097}) {  // float4 path, scalar path
      for (int num_blocks : {1, 3, 64}) {
        for (int num_threads : {64, 256, 512}) {
          CheckFinalSum({numel,
                         num_sources,
                         /*self_index=*/3,
                         /*scale=*/0.3f,
                         /*shard_offset=*/0,
                         /*recv_offset=*/0,
                         num_blocks,
                         num_threads});
          CheckIntraNodeSum({numel,
                             /*ranks_per_node=*/num_sources,
                             /*num_nodes=*/4,
                             /*rank_in_node=*/5,
                             /*grad_offset=*/0,
                             /*recv_offset=*/0,
                             num_blocks,
                             num_threads});
        }
      }
    }
  }
}

// Out-of-range layouts return cudaErrorInvalidValue without launching, so the
// output is not written and no CUDA error is left behind.
TEST(GinOverlapSumTest, InvalidLayoutsAreRejected) {
  const size_t numel = 16;
  const int max_sources = 4;
  const std::vector<float> self_values = RandomFloats(numel, 5);
  const std::vector<float> recv = RandomFloats(max_sources * numel, 6);
  DeviceFloats device_self(self_values, 0);
  DeviceFloats device_recv(recv, 0);

  struct SourceRange {
    int num_sources;
    int self_index;
  };
  for (SourceRange range : {SourceRange{0, 0},
                            SourceRange{-1, 0},
                            SourceRange{max_sources, -1},
                            SourceRange{max_sources, max_sources},
                            SourceRange{max_sources, max_sources + 1}}) {
    EXPECT_EQ(cudaErrorInvalidValue,
              LaunchFinalSum(device_self.data(),
                             device_recv.data(),
                             numel,
                             range.num_sources,
                             range.self_index,
                             /*scale=*/0.5f,
                             /*num_blocks=*/1,
                             /*stream=*/nullptr))
        << "num_sources=" << range.num_sources
        << " self_index=" << range.self_index;
  }

  SumLayout layout;
  layout.num_sources = max_sources;
  layout.self_index = 0;
  layout.numel = numel;
  layout.self_shard_stride = 0;
  layout.recv_row_stride = numel;
  layout.recv_shard_stride = 0;
  // LaunchSum<true, false, false> has no shards: num_shards must be 1.
  for (int num_shards : {0, 2}) {
    layout.num_shards = num_shards;
    EXPECT_EQ(cudaErrorInvalidValue,
              (LaunchSum<true, false, false>(device_self.data(),
                                             device_recv.data(),
                                             layout,
                                             /*scale=*/0.5f,
                                             /*num_blocks=*/1,
                                             /*stream=*/nullptr,
                                             /*num_threads=*/256)))
        << "num_shards=" << num_shards;
  }
  // LaunchSum<false, true, true> takes any num_shards >= 1.
  layout.num_shards = 0;
  EXPECT_EQ(cudaErrorInvalidValue,
            (LaunchSum<false, true, true>(device_self.data(),
                                          device_recv.data(),
                                          layout,
                                          /*scale=*/1.0f,
                                          /*num_blocks=*/1,
                                          /*stream=*/nullptr,
                                          /*num_threads=*/512)))
      << "num_shards=0";

  CHECK_CUDA(cudaDeviceSynchronize());
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
  ExpectSameBits(self_values, device_self.CopyToHost(), "self_values");
  ExpectSameBits(recv, device_recv.CopyToHost(), "recv");
}

}  // namespace distributed
}  // namespace paddle
