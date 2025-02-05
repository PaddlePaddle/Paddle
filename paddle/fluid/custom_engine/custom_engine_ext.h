// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#if !defined(_WIN32)
#include <cstddef>
#include <cstring>
#include <unordered_map>

#include "paddle/phi/backends/device_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct C_Operation_st* C_Operation;
typedef struct C_IrContext_st* C_IrContext;
typedef struct C_Value_st* C_Value;
typedef struct C_Place_st* C_Place;
typedef struct C_KernelKey_st* C_KernelKey;
typedef struct C_Block_st* C_Block;
typedef struct std::unordered_map<C_Operation_st, C_Operation_st>*
    C_Operation_Map;
typedef struct std::unordered_map<C_Value_st, C_Value_st>* C_Value_Map;
typedef struct C_CustomEngineInstruction_st* C_CustomEngineInstruction;

struct C_CustomEngineLowerParams {
  C_IrContext ir_context;
  C_Operation operation;
  C_KernelKey kernel_key;
  C_Place place;
  C_Operation_Map map_op_pair;
  C_Value_Map map_value_pair;
  C_Block block;
};

struct C_CustomEngineInterface {
  size_t size;
  C_Status (*register_custom_engine_op)();
  C_Status (*graph_engine_build)(C_CustomEngineInstruction);
  C_Status (*graph_engine_execute)(C_CustomEngineInstruction);
  C_Status (*custom_engine_op_lower)(C_CustomEngineLowerParams*);
};

struct CustomEngineParams {
  size_t size;
  C_CustomEngineInterface* interface;
};

// Plugin implement it and fill CustomEngineParams
void InitPluginCustomEngine(CustomEngineParams*);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
