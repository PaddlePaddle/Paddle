// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#ifndef _WIN32
#include <nccl.h>
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#endif
#include <cudnn.h>
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/cudnn_rnn_cache.h"
#endif

namespace paddle {
namespace framework {

namespace detail {

template <bool kStop, int kStart, int kEnd, typename T1, typename T2,
          typename... Args>
struct TypePosFinderImpl {
  static constexpr int kPos =
      std::is_same<T1, T2>::value
          ? kStart
          : TypePosFinderImpl<kStart + 2 == kEnd, kStart + 1, kEnd, T1,
                              Args...>::kPos;
};

template <int kStart, int kEnd, typename T1, typename T2>
struct TypePosFinderImpl<true, kStart, kEnd, T1, T2> {
  static constexpr int kPos = std::is_same<T1, T2>::value ? kStart : -1;
};

// TypePosFinder helps to find the position in which T is inside Args...
// If T is not inside Args..., kPos would be -1
template <typename T, typename... Args>
struct TypePosFinder {
  static constexpr int kPos =
      TypePosFinderImpl<sizeof...(Args) == 1, 0, sizeof...(Args), T,
                        Args...>::kPos;
};

template <typename... Args>
struct VarTypeRegistryImpl {
  static constexpr size_t kRegisteredTypeNum = sizeof...(Args);
  using ArgTuple = std::tuple<Args...>;

  // TypePos() returns the position in which T is inside Args...
  // If T is not inside Args... or T is void, return -1
  template <typename T>
  static constexpr int TypePos() {
    return std::is_same<T, void>::value ? -1 : TypePosFinder<T, Args...>::kPos;
  }

  // IsRegistered() returns whether T is registered inside RegistryImpl
  template <typename T>
  static constexpr bool IsRegistered() {
    return TypePos<T>() >= 0;
  }
};

}  // namespace detail

#define REG_PROTO_VAR_TYPE_TRAIT(type, proto_id)         \
  template <>                                            \
  struct VarTypeTrait<type> {                            \
    static_assert(VarTypeRegistry::IsRegistered<type>(), \
                  "Must be registered type");            \
    using Type = type;                                   \
    static constexpr int kId = proto_id;                 \
  }

/**
 * The following codes are designed to register variable types.
 * Only registered types can be stored in Variable.
 * This registry mechanism is designed to speed up Variable.
 */

// Users should add other variable types below.
// Paddle would generate unique Ids for each registered variable types.
class Scope;

using VarTypeRegistry = detail::VarTypeRegistryImpl<
    LoDTensor, SelectedRows, std::vector<Scope *>, LoDRankTable, LoDTensorArray,
    platform::PlaceList, ReaderHolder, Tensor, std::string, Scope *,
    std::map<size_t, Tensor>, operators::reader::LoDTensorBlockingQueueHolder,
    int, float,
#ifdef PADDLE_WITH_CUDA
#ifndef _WIN32
    ncclUniqueId, platform::Communicator,
#endif
    operators::AlgorithmsCache<cudnnConvolutionFwdAlgo_t>,
    operators::AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>,
    operators::AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>,
    operators::CudnnRNNCache,
#endif
    void>;  // void indicates end of registration, add other types before void

template <typename T>
struct VarTypeTrait {
  static_assert(std::is_same<T, void>::value ||
                    VarTypeRegistry::IsRegistered<T>(),
                "Must be registered type");
  using Type = T;
  // Default id generation
  static constexpr int kId = VarTypeRegistry::TypePos<T>() +
                             static_cast<int>(proto::VarType::TUPLE) * 2;
};

// Users should set some of variable type ids to be what is defined in
// framework.proto here
REG_PROTO_VAR_TYPE_TRAIT(LoDTensor, proto::VarType::LOD_TENSOR);
REG_PROTO_VAR_TYPE_TRAIT(SelectedRows, proto::VarType::SELECTED_ROWS);
REG_PROTO_VAR_TYPE_TRAIT(std::vector<Scope *>, proto::VarType::STEP_SCOPES);
REG_PROTO_VAR_TYPE_TRAIT(LoDRankTable, proto::VarType::LOD_RANK_TABLE);
REG_PROTO_VAR_TYPE_TRAIT(LoDTensorArray, proto::VarType::LOD_TENSOR_ARRAY);
REG_PROTO_VAR_TYPE_TRAIT(platform::PlaceList, proto::VarType::PLACE_LIST);
REG_PROTO_VAR_TYPE_TRAIT(ReaderHolder, proto::VarType::READER);

/** End of variable type registration */

// Besides register variable id, it is helpful to register a
// var_id -> std::type_index (for example, get var names according to id)
namespace detail {

template <int kStart, int kEnd, bool kStop>
struct VarIdToTypeIndexMapInitializerImpl {
  static void Init(std::unordered_map<int, std::type_index> *m) {
    using Type =
        typename std::tuple_element<kStart, VarTypeRegistry::ArgTuple>::type;
    constexpr int kId = VarTypeTrait<Type>::kId;
    if (!std::is_same<Type, void>::value) {
      m->emplace(kId, std::type_index(typeid(Type)));
    }
    VarIdToTypeIndexMapInitializerImpl<kStart + 1, kEnd,
                                       kStart + 1 == kEnd>::Init(m);
  }
};

template <int kStart, int kEnd>
struct VarIdToTypeIndexMapInitializerImpl<kStart, kEnd, true> {
  static void Init(std::unordered_map<int, std::type_index> *m) {}
};

// VarIdToTypeIndexMapInitializer is designed to initialize var_id ->
// std::type_index map
using VarIdToTypeIndexMapInitializer =
    VarIdToTypeIndexMapInitializerImpl<0, VarTypeRegistry::kRegisteredTypeNum,
                                       VarTypeRegistry::kRegisteredTypeNum ==
                                           0>;

struct VarIdToTypeIndexMapHolder {
 public:
  static const std::type_index &ToTypeIndex(int var_id) {
    static const VarIdToTypeIndexMapHolder instance;
    auto it = instance.var_type_map_.find(var_id);
    PADDLE_ENFORCE(it != instance.var_type_map_.end(),
                   "VarId %d is not registered.", var_id);
    return it->second;
  }

 private:
  VarIdToTypeIndexMapHolder() {
    VarIdToTypeIndexMapInitializer::Init(&var_type_map_);
  }
  std::unordered_map<int, std::type_index> var_type_map_;
};

}  // namespace detail

const char *ToTypeName(int var_id);
const std::type_index &ToTypeIndex(int var_id);

template <typename T>
inline constexpr bool IsRegisteredVarType() {
  return VarTypeRegistry::IsRegistered<T>();
}

#undef REG_PROTO_VAR_TYPE_TRAIT
}  // namespace framework
}  // namespace paddle
