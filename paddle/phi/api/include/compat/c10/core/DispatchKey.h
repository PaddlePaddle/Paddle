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

#pragma once

#include <c10/core/DeviceType.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace c10 {

#define C10_FORALL_BACKEND_COMPONENTS(_, extra) \
  _(CPU, extra)                                 \
  _(CUDA, extra)                                \
  _(HIP, extra)                                 \
  _(XLA, extra)                                 \
  _(MPS, extra)                                 \
  _(IPU, extra)                                 \
  _(XPU, extra)                                 \
  _(HPU, extra)                                 \
  _(VE, extra)                                  \
  _(Lazy, extra)                                \
  _(MTIA, extra)                                \
  _(MAIA, extra)                                \
  _(PrivateUse1, extra)                         \
  _(PrivateUse2, extra)                         \
  _(PrivateUse3, extra)                         \
  _(Meta, extra)

#define C10_FORALL_FUNCTIONALITY_KEYS(_) \
  _(Dense, ) /* NOLINT */                \
  _(Quantized, Quantized)                \
  _(Sparse, Sparse)                      \
  _(SparseCsr, SparseCsr)                \
  _(NestedTensor, NestedTensor)          \
  _(AutogradFunctionality, Autograd)

enum class BackendComponent : uint8_t {
  InvalidBit = 0,
#define DEFINE_BACKEND_COMPONENT(n, _) n##Bit,
  C10_FORALL_BACKEND_COMPONENTS(DEFINE_BACKEND_COMPONENT, unused)
#undef DEFINE_BACKEND_COMPONENT

      EndOfBackendKeys = MetaBit,
};

enum class DispatchKey : uint16_t {
  Undefined = 0,

  CatchAll = Undefined,

  Dense,

  FPGA,

  Vulkan,
  Metal,

  Quantized,

  CustomRNGKeyId,

  MkldnnCPU,

  Sparse,

  SparseCsr,

  NestedTensor,

  BackendSelect,

  Python,

  Fake,
  FuncTorchDynamicLayerBackMode,

  Functionalize,

  Named,

  Conjugate,

  Negative,

  ZeroTensor,

  ADInplaceOrView,

  AutogradOther,

  AutogradFunctionality,

  AutogradNestedTensor,

  Tracer,

  AutocastCPU,
  AutocastMTIA,
  AutocastMAIA,
  AutocastXPU,
  AutocastIPU,
  AutocastHPU,
  AutocastXLA,
  AutocastMPS,
  AutocastCUDA,
  AutocastPrivateUse1,

  FuncTorchBatched,

  BatchedNestedTensor,

  FuncTorchVmapMode,

  Batched,

  VmapMode,

  FuncTorchGradWrapper,

  DeferredInit,

  PythonTLSSnapshot,

  FuncTorchDynamicLayerFrontMode,

  TESTING_ONLY_GenericWrapper,

  TESTING_ONLY_GenericMode,

  PreDispatch,

  PythonDispatcher,

  EndOfFunctionalityKeys,

#define DEFINE_PER_BACKEND_KEYS_FOR_BACKEND(n, prefix) prefix##n,

#define DEFINE_PER_BACKEND_KEYS(fullname, prefix)                        \
  StartOf##fullname##Backends,                                           \
      C10_FORALL_BACKEND_COMPONENTS(DEFINE_PER_BACKEND_KEYS_FOR_BACKEND, \
                                    prefix) EndOf##fullname##Backends =  \
          prefix##Meta,

  C10_FORALL_FUNCTIONALITY_KEYS(DEFINE_PER_BACKEND_KEYS)

#undef DEFINE_PER_BACKEND_KEYS
#undef DEFINE_PER_BACKEND_KEYS_FOR_BACKEND

      EndOfRuntimeBackendKeys = EndOfAutogradFunctionalityBackends,

  Autograd,
  CompositeImplicitAutograd,

  FuncTorchBatchedDecomposition,
  CompositeImplicitAutogradNestedTensor,
  CompositeExplicitAutograd,
  CompositeExplicitAutogradNonFunctional,

  StartOfAliasKeys = Autograd,
  EndOfAliasKeys = CompositeExplicitAutogradNonFunctional,

  CPUTensorId = CPU,
  CUDATensorId = CUDA,
  DefaultBackend = CompositeExplicitAutograd,
  PrivateUse1_PreAutograd = AutogradPrivateUse1,
  PrivateUse2_PreAutograd = AutogradPrivateUse2,
  PrivateUse3_PreAutograd = AutogradPrivateUse3,
  Autocast = AutocastCUDA,
};

static_assert(
    (static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) +
     static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) <= 64,
    "The BackendComponent and DispatchKey enums (below EndOfFunctionalityKeys)"
    " both map to backend and functionality bits"
    " into a 64-bit bitmask; you must have less than 64 total entries between "
    "them");

constexpr bool isAliasDispatchKey(DispatchKey k) {
  return k >= DispatchKey::StartOfAliasKeys && k <= DispatchKey::EndOfAliasKeys;
}

constexpr bool isPerBackendFunctionalityKey(DispatchKey k) {
  if (k == DispatchKey::Dense || k == DispatchKey::Quantized ||
      k == DispatchKey::Sparse || k == DispatchKey::SparseCsr ||
      k == DispatchKey::AutogradFunctionality ||
      k == DispatchKey::NestedTensor) {
    return true;
  } else {
    return false;
  }
}

constexpr uint8_t num_functionality_keys =
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);

constexpr uint8_t num_backends =
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys);

static_assert(static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) <= 16,
              "BackendComponent currently only supports <= 16 backends. "
              "If we really need to extend this, "
              "there are a few places where this invariant is baked in");

constexpr uint8_t numPerBackendFunctionalityKeys() {
  uint8_t count = 0;
  for (uint8_t k = 0; k <= num_functionality_keys; ++k) {
    if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(k))) ++count;
  }
  return count;
}

#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
constexpr uint16_t num_runtime_entries = 8;
#else
constexpr uint16_t num_runtime_entries =
    num_functionality_keys +
    (numPerBackendFunctionalityKeys() * (num_backends - 1));
#endif

constexpr uint16_t full_backend_mask =
    (static_cast<uint16_t>(1) << num_backends) - 1;

const char* toString(DispatchKey /*t*/);
const char* toString(BackendComponent /*t*/);
std::ostream& operator<<(std::ostream& /*str*/, DispatchKey /*rhs*/);
std::ostream& operator<<(std::ostream& /*str*/, BackendComponent /*rhs*/);

DispatchKey getAutogradKeyFromBackend(BackendComponent k);

c10::DispatchKey parseDispatchKey(const std::string& k);

constexpr DispatchKey kAutograd = DispatchKey::Autograd;

constexpr BackendComponent toBackendComponent(DispatchKey k) {
  if (k >= DispatchKey::StartOfDenseBackends &&
      k <= DispatchKey::EndOfDenseBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends));
  } else if (k >= DispatchKey::StartOfQuantizedBackends &&
             k <= DispatchKey::EndOfQuantizedBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends));
  } else if (k >= DispatchKey::StartOfSparseBackends &&
             k <= DispatchKey::EndOfSparseBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends));
  } else if (k >= DispatchKey::StartOfSparseCsrBackends &&
             k <= DispatchKey::EndOfSparseCsrBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfSparseCsrBackends));
  } else if (k >= DispatchKey::StartOfNestedTensorBackends &&
             k <= DispatchKey::EndOfNestedTensorBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends));
  } else if (k >= DispatchKey::StartOfAutogradFunctionalityBackends &&
             k <= DispatchKey::EndOfAutogradFunctionalityBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(
            DispatchKey::StartOfAutogradFunctionalityBackends));
  } else {
    return BackendComponent::InvalidBit;
  }
}

constexpr DispatchKey toFunctionalityKey(DispatchKey k) {
  if (k <= DispatchKey::EndOfFunctionalityKeys) {
    return k;
  } else if (k <= DispatchKey::EndOfDenseBackends) {
    return DispatchKey::Dense;
  } else if (k <= DispatchKey::EndOfQuantizedBackends) {
    return DispatchKey::Quantized;
  } else if (k <= DispatchKey::EndOfSparseBackends) {
    return DispatchKey::Sparse;
  } else if (k <= DispatchKey::EndOfSparseCsrBackends) {
    return DispatchKey::SparseCsr;
  } else if (k <= DispatchKey::EndOfNestedTensorBackends) {
    return DispatchKey::NestedTensor;
  } else if (k <= DispatchKey::EndOfAutogradFunctionalityBackends) {
    return DispatchKey::AutogradFunctionality;
  } else {
    return DispatchKey::Undefined;
  }
}

BackendComponent toBackendComponent(DeviceType device_type);

constexpr DispatchKey toRuntimePerBackendFunctionalityKey(
    DispatchKey functionality_k, BackendComponent backend_k) {
  if (functionality_k == DispatchKey::Dense) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Sparse) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::SparseCsr) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfSparseCsrBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Quantized) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::NestedTensor) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::AutogradFunctionality) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(
            DispatchKey::StartOfAutogradFunctionalityBackends) +
        static_cast<uint8_t>(backend_k));
  }
  return DispatchKey::Undefined;
}

}  // namespace c10

namespace torch {
using c10::kAutograd;  // NOLINT
}  // namespace torch

namespace std {
template <>
struct hash<c10::DispatchKey> {
  typedef size_t result_type;
  typedef c10::DispatchKey argument_type;

  size_t operator()(c10::DispatchKey x) const { return static_cast<size_t>(x); }
};
}  // namespace std
