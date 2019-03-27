// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/multi_bin_buffered_allocator.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include "paddle/fluid/platform/lock_guard_ptr.h"

DEFINE_double(
    buffered_allocator_excess_times, 2,
    "Excess memory size times of buffered_allocator. BufferedAllocator"
    " would try to reuse memory freed previously, but the size of freed"
    " allocation may not be exactly the same as the requested. Here, we"
    " use a flag to control the excess times of reused memory size. "
    "Not quite sure what is the best excess times value.");

DEFINE_string(
    buffered_allocator_division_plan_path, "",
    "The file path which "
    "determines the memory size division plans of BufferedAllocator."
    "If it is empty, use the default division plan. The file must be a "
    "text file which each lines indicates the bound of division plan. "
    "For example, if the text file has 3 lines, which are '500M', '1G', "
    " '2G', the division plan would be [0, 500M), [500M, 1G), [1G, 2G) "
    "and [2G, +inf). Allocation request whose requested memory size is "
    "inside the last interval of division plan would be dispatched to "
    " underlying_allocator directly without caching when freed.");

namespace paddle {
namespace memory {
namespace allocation {

static std::string TrimStringAndToUpperCase(const std::string &str) {
  auto not_space = [](char ch) { return std::isspace(ch) == 0; };
  auto first_idx = static_cast<size_t>(
      std::find_if(str.begin(), str.end(), not_space) - str.begin());
  auto last_idx = static_cast<size_t>(
      std::find_if(str.rbegin(), str.rend(), not_space) - str.rbegin());
  if (first_idx == str.size() || last_idx == str.size()) return "";

  last_idx = str.size() - last_idx;
  auto ret = str.substr(first_idx, last_idx - first_idx);
  std::for_each(ret.begin(), ret.end(),
                [](char &ch) { ch = std::toupper(ch); });
  return ret;
}

namespace {

enum DivisionPlanFileStatus { kEOF, kException, kNormal };

}  // NOLINT

static size_t ParseStringToBytes(const std::string &original_str,
                                 DivisionPlanFileStatus *ret_code) {
  std::string str = TrimStringAndToUpperCase(original_str);

  if (str.empty()) {
    *ret_code = kEOF;
    return 0;
  }

  if (str.back() == 'B') {
    str.pop_back();
    if (str.empty()) {
      *ret_code = kException;
      return 0;
    }
  }

  size_t multiples = 1;
  switch (str.back()) {
    case 'G':
      multiples *= (static_cast<size_t>(1) << 30);
      break;
    case 'M':
      multiples *= (static_cast<size_t>(1) << 20);
      break;
    case 'K':
      multiples *= (static_cast<size_t>(1) << 10);
      break;
    default:
      break;
  }

  if (multiples != 1) {
    str.pop_back();
    if (str.empty()) {
      *ret_code = kException;
      return 0;
    }
  }

  str = TrimStringAndToUpperCase(str);
  double mem_val = -1.0;
  std::stringstream ss(str);
  if (!(ss >> mem_val) || mem_val < 0) {
    *ret_code = kException;
    return 0;
  }

  *ret_code = kNormal;
  return static_cast<size_t>(mem_val * multiples);
}

static std::string GetDebugStringOfPlan(const std::vector<size_t> &plan) {
  std::string ret("[");
  for (auto sz : plan) {
    ret += string::HumanReadableSize(sz);
    ret += ", ";
  }
  return ret + "]";
}

std::vector<size_t> ReadBufferedAllocatorDivisionPlanFromFile(
    const std::string &filepath) {
  std::ifstream is(filepath.c_str());
  PADDLE_ENFORCE(is.good(), "File %s not exist", filepath);
  std::string str;
  std::vector<size_t> plan;
  size_t line_num = 1;
  while (std::getline(is, str).good()) {
    DivisionPlanFileStatus status;
    size_t ret = ParseStringToBytes(str, &status);
    if (status == kEOF) {
      break;
    }
    if (status == kException) {
      PADDLE_THROW(
          "Invalid format in line %d of file %s: '%s'. Only support B, KB, MB, "
          "GB.",
          line_num, filepath, str);
    }
    plan.push_back(ret);
    ++line_num;
  }
  return plan;
}

static void CheckAndModifyMemoryDivisionPlan(
    std::vector<size_t> *division_plan) {
  // Check whether the division plan is strictly sorted
  bool is_strictly_sorted = true;
  for (size_t i = 1; i < division_plan->size(); ++i) {
    if ((*division_plan)[i - 1] >= (*division_plan)[i]) {
      is_strictly_sorted = false;
      break;
    }
  }
  PADDLE_ENFORCE(is_strictly_sorted, "Divison plan must be stricted sorted");

  // Insert 0 to disivion plan for clean binary searching code
  if (division_plan->empty() || division_plan->front() != 0) {
    division_plan->insert(division_plan->begin(), 0);
  }

  // Remove MAX from disivion plan for clean binary searching code
  constexpr auto kSizeTypeMax = std::numeric_limits<size_t>::max();
  if (division_plan->back() == kSizeTypeMax) {
    division_plan->pop_back();
  }

  PADDLE_ENFORCE(division_plan->size() >= 1, "Division plan cannot be empty");
}

static std::vector<size_t> GetDefaultDivisionPlan() {
  if (!FLAGS_buffered_allocator_division_plan_path.empty()) {
    return ReadBufferedAllocatorDivisionPlanFromFile(
        FLAGS_buffered_allocator_division_plan_path);
  }

  // Default division plan is 4K, 8K, 16K, ..., 500M, 1G
  constexpr size_t kMaxLogSize = 30;
  std::vector<size_t> plan;
  for (size_t i = 12; i <= kMaxLogSize; ++i) {
    plan.push_back(static_cast<size_t>(1) << i);
  }
  return plan;
}

inline static size_t FindDivisionPlanBinIndex(const std::vector<size_t> &bins,
                                              size_t size) {
  return static_cast<size_t>(std::upper_bound(bins.begin(), bins.end(), size) -
                             bins.begin() - 1);
}

inline static size_t TolerantUpperSize(size_t size) {
  return static_cast<size_t>(size * FLAGS_buffered_allocator_excess_times);
}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator)
    : MultiBinBufferedAllocator(std::move(underlying_allocator),
                                GetDefaultDivisionPlan()) {}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    const std::vector<size_t> &division_plan)
    : underlying_allocator_(std::move(underlying_allocator)),
      division_plan_(division_plan) {
  CheckAndModifyMemoryDivisionPlan(&division_plan_);
  allocations_.resize(division_plan_.size() - 1);
  accumulated_cache_size_.assign(division_plan_.size() - 1, 0UL);
  mtx_.resize(division_plan_.size() - 1);
  if (underlying_allocator_->IsAllocThreadSafe()) {
    for (auto &mtx : mtx_) {
      mtx.reset(new std::mutex());
    }
  }

  VLOG(1) << "Division plan is: " << GetDebugStringOfPlan(division_plan_);
  VLOG(1) << "FLAGS_buffered_allocator_excess_times = "
          << FLAGS_buffered_allocator_excess_times;
}

void MultiBinBufferedAllocator::FreeImpl(Allocation *allocation) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, allocation->size());
  if (bin_index < allocations_.size()) {
    platform::LockGuardPtr<std::mutex> guard(mtx_[bin_index]);
    allocations_[bin_index].emplace(allocation->size(),
                                    AllocationPtr(allocation));
    accumulated_cache_size_[bin_index] += allocation->size();
  } else {
    underlying_allocator_->Free(allocation);
  }
}

// Maybe we can design more flexible FreeCache strategy based on bin_index
// and require size.
size_t MultiBinBufferedAllocator::ClearCache() {
  size_t accumulated_size = 0;
  // FIXME(zjl): free the largest first when there is no extra
  for (size_t i = allocations_.size() - 1; i != static_cast<size_t>(-1); --i) {
    platform::LockGuardPtr<std::mutex> lock(mtx_[i]);
    allocations_[i].clear();
    accumulated_size += accumulated_cache_size_[i];
    accumulated_cache_size_[i] = 0;
  }
  return accumulated_size;
}

Allocation *MultiBinBufferedAllocator::AllocateImpl(size_t size, Attr attr) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, size);
  auto upper_size = TolerantUpperSize(size);

  for (; bin_index < allocations_.size() &&
         upper_size >= division_plan_[bin_index];
       ++bin_index) {
    auto &allocation = allocations_[bin_index];
    platform::LockGuardPtr<std::mutex> lock(mtx_[bin_index]);
    auto it = allocation.lower_bound(size);
    if (it != allocation.end() && it->second->size() <= upper_size) {
      size_t sz = it->second->size();
      auto ret = std::move(it->second);
      allocation.erase(it);
      accumulated_cache_size_[bin_index] -= sz;
      VLOG(3) << "Allocate " << sz << "(required " << size
              << ") from cache directly";
      return ret.release();
    }
  }

  size_t retry_time = 1;
  while (true) {
    try {
      auto ret = underlying_allocator_->Allocate(size, attr).release();
      VLOG(2) << "Allocate " << size << " from underlying directly";
      return ret;
    } catch (BadAlloc &) {
      size_t actual_free_size = ClearCache();
      VLOG(1) << retry_time << "-th free " << actual_free_size
              << " bytes caches";
      if (actual_free_size == 0) throw;
    }
    ++retry_time;
  }
}

void UseMultiBinBufferedAllocatorGFlags() {}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
