// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "glog/logging.h"
#include "paddle/phi/core/distributed/store/store.h"

namespace phi {
namespace distributed {

enum TraceEventType {
  TraceEventStart,
  TraceEventEnd,
};

using TraceMap =
    std::map<uint64_t, std::map<int, std::pair<std::string, TraceEventType>>>;

inline std::string GetTraceStartKey(const std::string& backend, int rank) {
  return backend + "_" + std::to_string(rank) + "_trace_start";
}

inline std::string GetTraceEndKey(const std::string& backend, int rank) {
  return backend + "_" + std::to_string(rank) + "_trace_end";
}

inline std::string GetExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exception_ptr) {
  if (exception_ptr == nullptr) {
    return "No exception found";
  }
  try {
    std::rethrow_exception(exception_ptr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

inline bool UpdateTraceMsg(std::shared_ptr<Store> store,
                           const std::string& key,
                           uint64_t seq,
                           const std::string& comm_type) {
  VLOG(1) << "update trace msg key: " << key << ", type " << comm_type
          << ", seq " << seq;
  std::vector<uint8_t> value(comm_type.size() + sizeof(seq) + 1);
  VLOG(1) << "update trace msg key: " << key << ", type " << comm_type
          << ", seq " << seq;
  memcpy(value.data(), &seq, sizeof(seq));
  VLOG(1) << "update trace msg key: " << key << ", type " << comm_type
          << ", seq " << seq;
  memcpy(value.data() + sizeof(seq), comm_type.data(), comm_type.size());
  VLOG(1) << "update trace msg key: " << key << ", type " << comm_type
          << ", seq " << seq;
  try {
    VLOG(0) << "update trace msg key: " << key << ", type " << comm_type
            << ", seq " << seq;
    store->set(key, value);
    VLOG(0) << "update trace msg key: " << key << ", type " << comm_type
            << ", seq " << seq;
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while updating trace msg, with seq: " << seq
               << ", key " << key;
    return false;
  }
}

inline bool ParseTraceValue(std::shared_ptr<Store> store,
                            const std::string& key,
                            uint64_t* seq,
                            std::string* comm_type) {
  try {
    std::vector<uint8_t> value = store->get(key);
    VLOG(0) << "parse trace key: " << key << ", val size: " << value.size();
    memcpy(seq, value.data(), sizeof(*seq));
    std::string type_value(
        reinterpret_cast<char*>(value.data() + sizeof(*seq)));
    *comm_type = type_value;
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while parsing trace value, with key: " << key;
    return false;
  }
}

inline std::string RanksToString(const std::vector<int>& ranks) {
  std::string result;
  for (int rank : ranks) {
    if (result.empty()) {
      result += std::to_string(rank);
    } else {
      result += ", " + std::to_string(rank);
    }
  }
  return result;
}

inline std::string AnalyzeTraceMsg(const TraceMap& trace_map) {
  uint64_t lag_seq = trace_map.begin()->first;
  std::vector<int> start_ranks;
  std::vector<int> end_ranks;
  for (auto& p : trace_map.begin()->second) {
    if (p.second.second == TraceEventStart) {
      start_ranks.emplace_back(p.first);
    } else {
      end_ranks.emplace_back(p.first);
    }
  }

  VLOG(0) << "debug start_ranks size: " << start_ranks.size()
          << ", end_ranks size: " << end_ranks.size();
  std::string result =
      "\n\t The lagging/dead/mismatched ranks that has desync problem are:";
  if (start_ranks.size()) {
    result +=
        "ranks [" + RanksToString(start_ranks) +
        "] joined but do not finish collective seq: " + std::to_string(lag_seq);
  }
  if (end_ranks.size()) {
    result += ", ranks [" + RanksToString(end_ranks) +
              "] finished collective seq: " + std::to_string(lag_seq) +
              ", but didnt join collective seq: " + std::to_string(lag_seq + 1);
  }
  VLOG(0) << "debug analyze trace result: " << result;
  return result;
}

inline std::string GenerateTraceMsg(std::shared_ptr<Store> store,
                                    const std::string& backend,
                                    int curr_rank,
                                    int world_size) {
  std::string result;
  TraceMap trace_map;

  uint64_t curr_seq;
  std::string curr_comm_type;

  VLOG(0) << "generate trace msg backend: " << backend;
  for (int rank = 0; rank < world_size; ++rank) {
    VLOG(0) << "generate trace msg rank: " << rank;
    uint64_t seq_start = 0;
    {
      std::string trace_start_key = GetTraceStartKey(backend, rank);
      if (!store->check(trace_start_key)) {
        VLOG(0) << "check empty trace msg key: " << trace_start_key;
        continue;
      }

      std::string comm_type;
      if (!ParseTraceValue(store, trace_start_key, &seq_start, &comm_type)) {
        return result;
      }
      VLOG(0) << "generate trace msg key: " << trace_start_key;
      trace_map[seq_start].emplace(rank,
                                   std::make_pair(comm_type, TraceEventStart));
      if (rank == curr_rank) {
        curr_seq = seq_start;
        curr_comm_type = std::move(comm_type);
      }
    }
    {
      std::string trace_end_key = GetTraceEndKey(backend, rank);
      if (!store->check(trace_end_key)) {
        VLOG(0) << "check empty trace msg key: " << trace_end_key;
        continue;
      }

      uint64_t seq = 0;
      std::string comm_type;
      VLOG(0) << "generate trace msg key: " << trace_end_key;
      if (!ParseTraceValue(store, trace_end_key, &seq, &comm_type)) {
        return result;
      }
      VLOG(0) << "generate trace msg key: " << trace_end_key;
      VLOG(0) << "generate trace msg seq_start: " << seq_start
              << ", seq: " << seq;
      if (seq == seq_start) {
        trace_map[seq][rank].second = TraceEventEnd;
        VLOG(0) << "generate trace seq: " << seq << ", rank: " << rank
                << ", event end";
      }
    }
  }
  VLOG(0) << "generate trace msg process result: " << result;
  result += "\n\t Global detail, rank: " + std::to_string(curr_rank) +
            " timeout at collective: " + curr_comm_type +
            ", seq: " + std::to_string(curr_seq);
  VLOG(0) << "generate trace msg process result: " << result;
  result += AnalyzeTraceMsg(trace_map);
  return result;
}

}  // namespace distributed
}  // namespace phi
