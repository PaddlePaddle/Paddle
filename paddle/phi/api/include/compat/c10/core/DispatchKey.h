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

inline BackendComponent toBackendComponent(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return BackendComponent::CPUBit;
    case DeviceType::CUDA:
      return BackendComponent::CUDABit;
    case DeviceType::XPU:
      return BackendComponent::XPUBit;
    default:
      return BackendComponent::InvalidBit;
  }
}

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

// toString implementations
inline const char* toString(DispatchKey t) {
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";
    case DispatchKey::Dense:
      return "Dense";
    case DispatchKey::FPGA:
      return "FPGA";
    case DispatchKey::Vulkan:
      return "Vulkan";
    case DispatchKey::Metal:
      return "Metal";
    case DispatchKey::Quantized:
      return "Quantized";
    case DispatchKey::CustomRNGKeyId:
      return "CustomRNGKeyId";
    case DispatchKey::MkldnnCPU:
      return "MkldnnCPU";
    case DispatchKey::Sparse:
      return "Sparse";
    case DispatchKey::SparseCsr:
      return "SparseCsr";
    case DispatchKey::NestedTensor:
      return "NestedTensor";
    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::Python:
      return "Python";
    case DispatchKey::Fake:
      return "Fake";
    case DispatchKey::FuncTorchDynamicLayerBackMode:
      return "FuncTorchDynamicLayerBackMode";
    case DispatchKey::Functionalize:
      return "Functionalize";
    case DispatchKey::Named:
      return "Named";
    case DispatchKey::Conjugate:
      return "Conjugate";
    case DispatchKey::Negative:
      return "Negative";
    case DispatchKey::ZeroTensor:
      return "ZeroTensor";
    case DispatchKey::ADInplaceOrView:
      return "ADInplaceOrView";
    case DispatchKey::AutogradOther:
      return "AutogradOther";
    case DispatchKey::AutogradFunctionality:
      return "AutogradFunctionality";
    case DispatchKey::AutogradNestedTensor:
      return "AutogradNestedTensor";
    case DispatchKey::Tracer:
      return "Tracer";
    case DispatchKey::AutocastCPU:
      return "AutocastCPU";
    case DispatchKey::AutocastMTIA:
      return "AutocastMTIA";
    case DispatchKey::AutocastMAIA:
      return "AutocastMAIA";
    case DispatchKey::AutocastXPU:
      return "AutocastXPU";
    case DispatchKey::AutocastIPU:
      return "AutocastIPU";
    case DispatchKey::AutocastHPU:
      return "AutocastHPU";
    case DispatchKey::AutocastXLA:
      return "AutocastXLA";
    case DispatchKey::AutocastMPS:
      return "AutocastMPS";
    case DispatchKey::AutocastCUDA:
      return "AutocastCUDA";
    case DispatchKey::AutocastPrivateUse1:
      return "AutocastPrivateUse1";
    case DispatchKey::FuncTorchBatched:
      return "FuncTorchBatched";
    case DispatchKey::BatchedNestedTensor:
      return "BatchedNestedTensor";
    case DispatchKey::FuncTorchVmapMode:
      return "FuncTorchVmapMode";
    case DispatchKey::Batched:
      return "Batched";
    case DispatchKey::VmapMode:
      return "VmapMode";
    case DispatchKey::FuncTorchGradWrapper:
      return "FuncTorchGradWrapper";
    case DispatchKey::DeferredInit:
      return "DeferredInit";
    case DispatchKey::PythonTLSSnapshot:
      return "PythonTLSSnapshot";
    case DispatchKey::FuncTorchDynamicLayerFrontMode:
      return "FuncTorchDynamicLayerFrontMode";
    case DispatchKey::TESTING_ONLY_GenericWrapper:
      return "TESTING_ONLY_GenericWrapper";
    case DispatchKey::TESTING_ONLY_GenericMode:
      return "TESTING_ONLY_GenericMode";
    case DispatchKey::PreDispatch:
      return "PreDispatch";
    case DispatchKey::PythonDispatcher:
      return "PythonDispatcher";
    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::CompositeImplicitAutograd:
      return "CompositeImplicitAutograd";
    case DispatchKey::FuncTorchBatchedDecomposition:
      return "FuncTorchBatchedDecomposition";
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      return "CompositeImplicitAutogradNestedTensor";
    case DispatchKey::CompositeExplicitAutograd:
      return "CompositeExplicitAutograd";
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      return "CompositeExplicitAutogradNonFunctional";
    default:
      return "Unknown";
  }
}

inline const char* toString(BackendComponent t) {
  switch (t) {
    case BackendComponent::CPUBit:
      return "CPUBit";
    case BackendComponent::CUDABit:
      return "CUDABit";
    case BackendComponent::HIPBit:
      return "HIPBit";
    case BackendComponent::XLABit:
      return "XLABit";
    case BackendComponent::MPSBit:
      return "MPSBit";
    case BackendComponent::IPUBit:
      return "IPUBit";
    case BackendComponent::XPUBit:
      return "XPUBit";
    case BackendComponent::HPUBit:
      return "HPUBit";
    case BackendComponent::VEBit:
      return "VEBit";
    case BackendComponent::LazyBit:
      return "LazyBit";
    case BackendComponent::MTIABit:
      return "MTIABit";
    case BackendComponent::MAIABit:
      return "MAIABit";
    case BackendComponent::PrivateUse1Bit:
      return "PrivateUse1Bit";
    case BackendComponent::PrivateUse2Bit:
      return "PrivateUse2Bit";
    case BackendComponent::PrivateUse3Bit:
      return "PrivateUse3Bit";
    case BackendComponent::MetaBit:
      return "MetaBit";
    default:
      return "InvalidBit";
  }
}

// operator<< implementations
inline std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

inline std::ostream& operator<<(std::ostream& str, BackendComponent rhs) {
  return str << toString(rhs);
}

// getAutogradKeyFromBackend implementation
inline DispatchKey getAutogradKeyFromBackend(BackendComponent k) {
  switch (k) {
    case BackendComponent::CPUBit:
      return DispatchKey::AutogradCPU;
    case BackendComponent::CUDABit:
      return DispatchKey::AutogradCUDA;
    case BackendComponent::XPUBit:
      return DispatchKey::AutogradXPU;
    case BackendComponent::IPUBit:
      return DispatchKey::AutogradIPU;
    case BackendComponent::HPUBit:
      return DispatchKey::AutogradHPU;
    case BackendComponent::LazyBit:
      return DispatchKey::AutogradLazy;
    case BackendComponent::MetaBit:
      return DispatchKey::AutogradMeta;
    case BackendComponent::MPSBit:
      return DispatchKey::AutogradMPS;
    case BackendComponent::PrivateUse1Bit:
      return DispatchKey::AutogradPrivateUse1;
    case BackendComponent::PrivateUse2Bit:
      return DispatchKey::AutogradPrivateUse2;
    case BackendComponent::PrivateUse3Bit:
      return DispatchKey::AutogradPrivateUse3;
    default:
      return DispatchKey::AutogradOther;
  }
}

// parseDispatchKey implementation
inline c10::DispatchKey parseDispatchKey(const std::string& k) {
  if (k == "Undefined") return DispatchKey::Undefined;
  if (k == "Dense") return DispatchKey::Dense;
  if (k == "FPGA") return DispatchKey::FPGA;
  if (k == "Vulkan") return DispatchKey::Vulkan;
  if (k == "Metal") return DispatchKey::Metal;
  if (k == "Quantized") return DispatchKey::Quantized;
  if (k == "Sparse") return DispatchKey::Sparse;
  if (k == "SparseCsr") return DispatchKey::SparseCsr;
  if (k == "NestedTensor") return DispatchKey::NestedTensor;
  if (k == "BackendSelect") return DispatchKey::BackendSelect;
  if (k == "Python") return DispatchKey::Python;
  if (k == "Fake") return DispatchKey::Fake;
  if (k == "Functionalize") return DispatchKey::Functionalize;
  if (k == "Named") return DispatchKey::Named;
  if (k == "Conjugate") return DispatchKey::Conjugate;
  if (k == "Negative") return DispatchKey::Negative;
  if (k == "ZeroTensor") return DispatchKey::ZeroTensor;
  if (k == "ADInplaceOrView") return DispatchKey::ADInplaceOrView;
  if (k == "AutogradOther") return DispatchKey::AutogradOther;
  if (k == "AutogradFunctionality") return DispatchKey::AutogradFunctionality;
  if (k == "AutogradNestedTensor") return DispatchKey::AutogradNestedTensor;
  if (k == "Tracer") return DispatchKey::Tracer;
  if (k == "AutocastCPU") return DispatchKey::AutocastCPU;
  if (k == "AutocastCUDA") return DispatchKey::AutocastCUDA;
  if (k == "Autograd") return DispatchKey::Autograd;
  if (k == "CompositeImplicitAutograd")
    return DispatchKey::CompositeImplicitAutograd;
  if (k == "CompositeExplicitAutograd")
    return DispatchKey::CompositeExplicitAutograd;
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
