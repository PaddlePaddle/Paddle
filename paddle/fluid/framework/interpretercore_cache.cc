// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/interpretercore_cache.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

static InterpreterCoreInfoCache &InterpreterCoreInfoCache::Instance() {
  static InterpreterCoreInfoCache g_info_cache;
  return g_info_cache;
}

// Create a sub program form promgram_desc by [start_op_index, end_op_index].
// in dy2static, origin program include forward program and backward program,
// forward op use [end_op_index+ (out_grad.size() * 2,  global_block->OpSize()]
CacheInfo GetInterpreterCoreInfoFromCache(const ProgramDesc &program_desc,
                                          const platform::Place &place,
                                          bool is_grad,
                                          int64_t program_id,
                                          framework::Scope *scope) {
  auto &interpretercore_info_cache =
      framework::InterpreterCoreInfoCache::Instance();
  // 1. Check whether has cached exe
  if (!interpretercore_info_cache.Has(program_id, is_grad)) {
    VLOG(1) << "No interpretercore from cache by: <program_id: " << program_id
            << ", is_grad: " << is_grad << ">.";

    if (interpretercore_info_cache.Size() > 4u /* max_cached_size*/) {
      VLOG(2) << "The cached_info size has exceeded max_cached_size: 4, so "
                 "clear all cache!";
      interpretercore_info_cache.Finalize();
    }
    // 2. Create a new exe.
    VLOG(1) << "Create a new interpretercore for: <program_id: " << program_id
            << ", is_grad: " << is_grad << ">.";
    auto core = std::make_shared<InterpreterCore>(
        place,
        program_desc->Block(0),
        /*skip_gc_vars=*/std::set<std::string>(),
        scope);
    auto core_and_program = std::make_pair(core, program_desc);

    // 3. Insert exe into cached map.
    auto &cached_value =
        interpretercore_info_cache.GetMutable(program_id, is_grad);
    cached_value.core_ = core_and_program.first;
    cached_value.program_ = core_and_program.second;
    return std::make_pair(core_and_program.first, /*is_new_created=*/true);

  } else {
    // 4. Get cached exe for cache.
    VLOG(1) << "Get a interpretercore from cache by: <program_id: "
            << program_id << ", is_grad: " << is_grad << ">.";
    auto &cached_value =
        interpretercore_info_cache.GetMutable(program_id, is_grad);

    auto &core = cached_value.core_;

    return std::make_pair(core, /*is_new_created=*/false);
  }
}

}  // namespace framework
}  // namespace paddle
