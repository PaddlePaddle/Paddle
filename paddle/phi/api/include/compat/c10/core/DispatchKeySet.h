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
#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <array>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace c10 {

struct FunctionalityOffsetAndMask {
  FunctionalityOffsetAndMask() = default;
  FunctionalityOffsetAndMask(uint16_t offset, uint16_t mask)
      : offset(offset), mask(mask) {}
  uint16_t offset{};
  uint16_t mask{};
};
static_assert(
    c10::num_runtime_entries < 65536,
    "The dispatcher currently only supports up to 2^16 runtime entries");

std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks();

static const std::array<FunctionalityOffsetAndMask, num_functionality_keys>&
offsetsAndMasks() {
  static auto offsets_and_masks_ = initializeFunctionalityOffsetsAndMasks();
  return offsets_and_masks_;
}

class DispatchKeySet final {
 public:
  enum Full { FULL };
  enum FullAfter { FULL_AFTER };
  enum Raw { RAW };

  constexpr DispatchKeySet() = default;

  constexpr DispatchKeySet(Full /*unused*/)
      : repr_((1ULL << (num_backends + num_functionality_keys - 1)) - 1) {}

  constexpr DispatchKeySet(FullAfter /*unused*/, DispatchKey t)
      : repr_((1ULL << (num_backends +
                        static_cast<uint8_t>(toFunctionalityKey(t)) - 1)) -
              1) {
    *this = add(DispatchKey::PythonDispatcher);
  }

  constexpr DispatchKeySet(Raw /*unused*/, uint64_t x) : repr_(x) {}

  constexpr explicit DispatchKeySet(BackendComponent k) {
    if (k == BackendComponent::InvalidBit) {
      repr_ = 0;
    } else {
      repr_ = 1ULL << (static_cast<uint8_t>(k) - 1);
    }
  }

  constexpr explicit DispatchKeySet(DispatchKey k) {  // NOLINT
    if (k == DispatchKey::Undefined) {
      repr_ = 0;
    } else if (k <= DispatchKey::EndOfFunctionalityKeys) {
      uint64_t functionality_val =
          1ULL << (num_backends + static_cast<uint8_t>(k) - 1);
      repr_ = functionality_val;
    } else if (k <= DispatchKey::EndOfRuntimeBackendKeys) {
      auto functionality_k = toFunctionalityKey(k);
      uint64_t functionality_val =
          1ULL << (num_backends + static_cast<uint8_t>(functionality_k) - 1);

      auto backend_k = toBackendComponent(k);
      uint64_t backend_val = backend_k == BackendComponent::InvalidBit
                                 ? 0
                                 : 1ULL
                                       << (static_cast<uint8_t>(backend_k) - 1);
      repr_ = functionality_val + backend_val;
    } else {
      repr_ = 0;
    }
  }

  constexpr uint64_t keys_to_repr(std::initializer_list<DispatchKey> ks) {
    uint64_t repr = 0;
    for (auto k : ks) {
      repr |= DispatchKeySet(k).repr_;
    }
    return repr;
  }

  constexpr uint64_t backend_bits_to_repr(
      std::initializer_list<BackendComponent> ks) {
    uint64_t repr = 0;
    for (auto k : ks) {
      repr |= DispatchKeySet(k).repr_;
    }
    return repr;
  }

  explicit constexpr DispatchKeySet(std::initializer_list<DispatchKey> ks)
      : repr_(keys_to_repr(ks)) {}

  explicit constexpr DispatchKeySet(std::initializer_list<BackendComponent> ks)
      : repr_(backend_bits_to_repr(ks)) {}

  inline bool has(DispatchKey t) const {
    PD_CHECK(t != DispatchKey::Undefined);
    return has_all(DispatchKeySet(t));
  }
  constexpr bool has_backend(BackendComponent t) const {
    return has_all(DispatchKeySet(t));
  }

  constexpr bool has_all(DispatchKeySet ks) const {
    return static_cast<bool>((repr_ & ks.repr_) == ks.repr_);
  }

  inline bool has_any(DispatchKeySet ks) const {
    PD_CHECK(((ks.repr_ & full_backend_mask) == 0) ||
             ((ks & DispatchKeySet({
                                       DispatchKey::Dense,
                                       DispatchKey::Quantized,
                                       DispatchKey::Sparse,
                                       DispatchKey::SparseCsr,
                                       DispatchKey::AutogradFunctionality,
                                   })
                        .repr_) == 0));
    return static_cast<bool>((repr_ & ks.repr_) != 0);
  }
  bool isSupersetOf(DispatchKeySet ks) const {
    return (repr_ & ks.repr_) == ks.repr_;
  }
  constexpr DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }
  constexpr DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }
  constexpr DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & (full_backend_mask | ~other.repr_));
  }

  constexpr DispatchKeySet operator^(DispatchKeySet other) const {
    return DispatchKeySet(repr_ ^ other.repr_);
  }
  bool operator==(DispatchKeySet other) const { return repr_ == other.repr_; }
  bool operator!=(DispatchKeySet other) const { return repr_ != other.repr_; }
  [[nodiscard]] constexpr DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
  [[nodiscard]] constexpr DispatchKeySet add(DispatchKeySet ks) const {
    return *this | ks;
  }

  [[nodiscard]] constexpr DispatchKeySet remove(DispatchKey t) const {
    return DispatchKeySet(repr_ &
                          ~(DispatchKeySet(t).repr_ & ~full_backend_mask));
  }
  constexpr DispatchKeySet remove_backend(BackendComponent b) const {
    return DispatchKeySet(repr_ & ~(DispatchKeySet(b).repr_));
  }
  bool empty() const { return repr_ == 0; }
  uint64_t raw_repr() const { return repr_; }

  static DispatchKeySet from_raw_repr(uint64_t x) {
    return DispatchKeySet(RAW, x);
  }

  DispatchKey highestFunctionalityKey() const {
    auto functionality_idx = indexOfHighestBit();
    if (functionality_idx < num_backends) return DispatchKey::Undefined;
    return static_cast<DispatchKey>(functionality_idx - num_backends);
  }

  BackendComponent highestBackendKey() const {
    auto backend_idx =
        DispatchKeySet(repr_ & full_backend_mask).indexOfHighestBit();
    if (backend_idx == 0) return BackendComponent::InvalidBit;
    return static_cast<BackendComponent>(backend_idx);
  }

  DispatchKey highestPriorityTypeId() const {
    auto functionality_k = highestFunctionalityKey();
    if (isPerBackendFunctionalityKey(functionality_k)) {
      return toRuntimePerBackendFunctionalityKey(functionality_k,
                                                 highestBackendKey());
    }
    return functionality_k;
  }

  uint8_t indexOfHighestBit() const {
    // Use compiler built-in instead of llvm::countLeadingZeros.
    if (repr_ == 0) return 0;
#if defined(_MSC_VER)
    unsigned long index;  // NOLINT(runtime/int)
    _BitScanReverse64(&index, repr_);
    return static_cast<uint8_t>(index + 1);
#else
    return static_cast<uint8_t>(64 - __builtin_clzll(repr_));
#endif
  }

#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
  /**
   * The method below maps the dispatch key in the enum DispatchKey to an
   * integer index in the dispatchTable_ array in OperatorEntry. The array
   * is trimmed for mobile to reduce peak memory usage since it's
   * unnecessary to reserve additional space for dispatch keys that will
   * never be used on mobile.
   */
  int getDispatchTableIndexForDispatchKeySet() const {
    auto dk = highestPriorityTypeId();
    switch (dk) {
      case DispatchKey::Undefined:
        return 0;
      case DispatchKey::CPU:
        return 1;
      case DispatchKey::QuantizedCPU:
        return 2;
      case DispatchKey::SparseCPU:
        return 3;
      case DispatchKey::BackendSelect:
        return 4;
      case DispatchKey::ADInplaceOrView:
        return 5;
      case DispatchKey::AutogradOther:
        return 6;
      case DispatchKey::AutogradCPU:
        return 7;
      default:
        return -1;
    }
  }
#else
  int getDispatchTableIndexForDispatchKeySet() const {
    auto functionality_idx =
        DispatchKeySet(repr_ >> num_backends).indexOfHighestBit();
    auto offset_and_mask = offsetsAndMasks()[functionality_idx];
    auto backend_idx =
        DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
    return offset_and_mask.offset + backend_idx;
  }
#endif

  uint64_t getBackendIndex() const {
    return DispatchKeySet((repr_ & full_backend_mask) >> 1).indexOfHighestBit();
  }

 private:
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;

 public:
  class iterator {
   public:
    using self_type = iterator;
    using iterator_category = std::input_iterator_tag;
    using value_type = DispatchKey;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    static const uint8_t end_iter_mask_val =
        num_backends + num_functionality_keys;
    static const uint8_t end_iter_key_val = num_functionality_keys;

    explicit iterator(const uint64_t* data_ptr,
                      uint8_t next_functionality = num_backends,
                      uint8_t next_backend = 0)
        : data_ptr_(data_ptr),
          next_functionality_(next_functionality),
          next_backend_(next_backend),
          current_dispatchkey_idx_(end_iter_key_val),
          current_backendcomponent_idx_(end_iter_key_val) {
      TORCH_INTERNAL_ASSERT(next_functionality_ >= num_backends,
                            "num_backends=",
                            static_cast<uint32_t>(num_backends),
                            "next_functionality_=",
                            static_cast<uint32_t>(next_functionality_));
      ++(*this);
    }

    self_type& operator++() {
      while (next_functionality_ < end_iter_mask_val) {
        if (*data_ptr_ & (1ULL << next_functionality_)) {
          current_dispatchkey_idx_ =
              static_cast<uint8_t>(next_functionality_ - num_backends);
          if (isPerBackendFunctionalityKey(
                  static_cast<DispatchKey>(current_dispatchkey_idx_))) {
            while (next_backend_ < num_backends) {
              if (*data_ptr_ & (1ULL << next_backend_)) {
                // BackendComponent is 1-based (InvalidBit=0, CPUBit=1, ...),
                // so bit position next_backend_ maps to enum value
                // next_backend_+1.
                current_backendcomponent_idx_ = next_backend_ + 1;
                ++next_backend_;
                return *this;
              }
              ++next_backend_;
            }
            // No backend bits set for this functionality key; advance.
            next_backend_ = 0;
            current_backendcomponent_idx_ = end_iter_key_val;
            ++next_functionality_;
            continue;
          }
          ++next_functionality_;
          return *this;
        }
        ++next_functionality_;
      }
      current_dispatchkey_idx_ = end_iter_key_val;
      current_backendcomponent_idx_ = end_iter_key_val;
      return *this;
    }

    self_type operator++(int) {
      self_type previous_iterator = *this;
      ++(*this);
      return previous_iterator;
    }

    bool operator==(const self_type& rhs) const {
      return next_functionality_ == rhs.next_functionality_ &&
             current_dispatchkey_idx_ == rhs.current_dispatchkey_idx_ &&
             next_backend_ == rhs.next_backend_ &&
             current_backendcomponent_idx_ == rhs.current_backendcomponent_idx_;
    }
    bool operator!=(const self_type& rhs) const {
      return next_functionality_ != rhs.next_functionality_ ||
             current_dispatchkey_idx_ != rhs.current_dispatchkey_idx_ ||
             next_backend_ != rhs.next_backend_ ||
             current_backendcomponent_idx_ != rhs.current_backendcomponent_idx_;
    }
    DispatchKey operator*() const {
      auto functionality_key =
          static_cast<DispatchKey>(current_dispatchkey_idx_);
      if (isPerBackendFunctionalityKey(functionality_key)) {
        auto next_key = toRuntimePerBackendFunctionalityKey(
            functionality_key,
            static_cast<BackendComponent>(current_backendcomponent_idx_));
        TORCH_INTERNAL_ASSERT(
            toBackendComponent(next_key) ==
                static_cast<BackendComponent>(current_backendcomponent_idx_),
            "Tried to map functionality key ",
            toString(functionality_key),
            " and backend bit ",
            toString(
                static_cast<BackendComponent>(current_backendcomponent_idx_)),
            " to a runtime key, but ended up with ",
            toString(next_key),
            ". This can happen if the order of the backend dispatch keys in "
            "DispatchKey.h isn't consistent.",
            " Please double check that enum for inconsistencies.");
        return next_key;
      } else {
        return functionality_key;
      }
    }

   private:
    const uint64_t* data_ptr_;
    uint8_t next_functionality_;
    uint8_t next_backend_;
    uint8_t current_dispatchkey_idx_;
    uint8_t current_backendcomponent_idx_;
  };

 public:
  iterator begin() const { return iterator(&repr_); }

  iterator end() const { return iterator(&repr_, iterator::end_iter_mask_val); }
};

std::string toString(DispatchKeySet /*ts*/);
std::ostream& operator<<(std::ostream& /*os*/, DispatchKeySet /*ts*/);

inline int getDispatchTableIndexForDispatchKey(DispatchKey k) {
  return DispatchKeySet(k).getDispatchTableIndexForDispatchKeySet();
}

constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutogradFunctionality,
    DispatchKey::AutogradOther,
    DispatchKey::AutogradNestedTensor,
});

constexpr DispatchKeySet autocast_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastMPS,
    DispatchKey::AutocastCUDA,
    DispatchKey::AutocastXPU,
    DispatchKey::AutocastIPU,
    DispatchKey::AutocastHPU,
    DispatchKey::AutocastXLA,
    DispatchKey::AutocastPrivateUse1,
    DispatchKey::AutocastMTIA,
    DispatchKey::AutocastMAIA,
});

constexpr DispatchKeySet default_included_set = DispatchKeySet({
    DispatchKey::BackendSelect,
    DispatchKey::ADInplaceOrView,
});

constexpr DispatchKeySet default_excluded_set = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastMPS,
    DispatchKey::AutocastCUDA,
    DispatchKey::AutocastXPU,
    DispatchKey::AutocastIPU,
    DispatchKey::AutocastHPU,
    DispatchKey::AutocastXLA,
    DispatchKey::AutocastPrivateUse1,
    DispatchKey::AutocastMTIA,
    DispatchKey::AutocastMAIA,
});

constexpr DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView =
    autograd_dispatch_keyset | DispatchKeySet(DispatchKey::ADInplaceOrView);

constexpr DispatchKeySet python_ks = DispatchKeySet({
    DispatchKey::Python,
    DispatchKey::PythonTLSSnapshot,
});

constexpr DispatchKeySet sparse_ks = DispatchKeySet(DispatchKey::Sparse);

constexpr DispatchKeySet sparse_csr_ks = DispatchKeySet(DispatchKey::SparseCsr);

constexpr DispatchKeySet mkldnn_ks = DispatchKeySet(DispatchKey::MkldnnCPU);

constexpr DispatchKeySet autogradother_backends =
    DispatchKeySet({DispatchKey::FPGA,
                    DispatchKey::Vulkan,
                    DispatchKey::Metal,
                    DispatchKey::CustomRNGKeyId,
                    DispatchKey::MkldnnCPU,
                    DispatchKey::Sparse,
                    DispatchKey::SparseCsr,
                    DispatchKey::Quantized}) |
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

constexpr DispatchKeySet after_autograd_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);

constexpr DispatchKeySet after_ADInplaceOrView_keyset = DispatchKeySet(
    DispatchKeySet::FULL_AFTER, c10::DispatchKey::ADInplaceOrView);

constexpr DispatchKeySet after_func_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::Functionalize)
        .remove(c10::DispatchKey::ADInplaceOrView);

constexpr DispatchKeySet backend_bitset_mask =
    DispatchKeySet(DispatchKeySet::RAW, (1ULL << num_backends) - 1);

constexpr auto inplace_or_view_ks =
    DispatchKeySet(DispatchKey::ADInplaceOrView);
constexpr auto autograd_cpu_ks = DispatchKeySet(DispatchKey::AutogradCPU);
constexpr auto autograd_ipu_ks = DispatchKeySet(DispatchKey::AutogradIPU);
constexpr auto autograd_mtia_ks = DispatchKeySet(DispatchKey::AutogradMTIA);
constexpr auto autograd_maia_ks = DispatchKeySet(DispatchKey::AutogradMAIA);
constexpr auto autograd_xpu_ks = DispatchKeySet(DispatchKey::AutogradXPU);
constexpr auto autograd_cuda_ks = DispatchKeySet(DispatchKey::AutogradCUDA);
constexpr auto autograd_xla_ks = DispatchKeySet(DispatchKey::AutogradXLA);
constexpr auto autograd_lazy_ks = DispatchKeySet(DispatchKey::AutogradLazy);
constexpr auto autograd_meta_ks = DispatchKeySet(DispatchKey::AutogradMeta);
constexpr auto autograd_mps_ks = DispatchKeySet(DispatchKey::AutogradMPS);
constexpr auto autograd_hpu_ks = DispatchKeySet(DispatchKey::AutogradHPU);
constexpr auto autograd_privateuse1_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse1);
constexpr auto autograd_privateuse2_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse2);
constexpr auto autograd_privateuse3_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse3);
constexpr auto autograd_other_ks = DispatchKeySet(DispatchKey::AutogradOther);
constexpr auto autograd_nested =
    DispatchKeySet(DispatchKey::AutogradNestedTensor);
constexpr auto functorch_transforms_ks =
    DispatchKeySet({DispatchKey::FuncTorchBatched,
                    DispatchKey::FuncTorchVmapMode,
                    DispatchKey::Batched,
                    DispatchKey::VmapMode,
                    DispatchKey::FuncTorchGradWrapper});

constexpr auto functorch_batched_ks =
    DispatchKeySet({DispatchKey::FuncTorchBatched});

constexpr DispatchKeySet backend_functionality_keys =
    DispatchKeySet({
        DispatchKey::Dense,
        DispatchKey::Quantized,
        DispatchKey::Sparse,
        DispatchKey::SparseCsr,
    }) |
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

struct OpTableOffsetAndMask {
  uint16_t offset;
  uint16_t backend_mask;
};

static_assert(num_backends <= 16,
              "Right now we expect the number of backends not to exceed 16. In "
              "the (unlikely) event"
              " that this changes, the size of "
              "OpTableOffsetAndMask::backend_mask needs to be increased too.");

bool isBackendDispatchKey(DispatchKey t);

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t);

bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k);

DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t);

inline DispatchKeySet getAutogradRelatedKeySetFromBackend(BackendComponent t) {
  switch (t) {
    case BackendComponent::CPUBit:
      return inplace_or_view_ks | autograd_cpu_ks;
    case BackendComponent::IPUBit:
      return inplace_or_view_ks | autograd_ipu_ks;
    case BackendComponent::MTIABit:
      return inplace_or_view_ks | autograd_mtia_ks;
    case BackendComponent::MAIABit:
      return inplace_or_view_ks | autograd_maia_ks;
    case BackendComponent::XPUBit:
      return inplace_or_view_ks | autograd_xpu_ks;
    case BackendComponent::CUDABit:
      return inplace_or_view_ks | autograd_cuda_ks;
    case BackendComponent::XLABit:
      return inplace_or_view_ks | autograd_xla_ks;
    case BackendComponent::LazyBit:
      return inplace_or_view_ks | autograd_lazy_ks;
    case BackendComponent::MetaBit:
      return inplace_or_view_ks | autograd_meta_ks;
    case BackendComponent::MPSBit:
      return inplace_or_view_ks | autograd_mps_ks;
    case BackendComponent::HPUBit:
      return inplace_or_view_ks | autograd_hpu_ks;
    case BackendComponent::PrivateUse1Bit:
      return inplace_or_view_ks | autograd_privateuse1_ks;
    case BackendComponent::PrivateUse2Bit:
      return inplace_or_view_ks | autograd_privateuse2_ks;
    case BackendComponent::PrivateUse3Bit:
      return inplace_or_view_ks | autograd_privateuse3_ks;
    default:
      return inplace_or_view_ks | autograd_other_ks;
  }
}

inline DispatchKeySet getAutocastRelatedKeySetFromBackend(BackendComponent t) {
  constexpr auto autocast_cpu_ks = DispatchKeySet(DispatchKey::AutocastCPU);
  constexpr auto autocast_mtia_ks = DispatchKeySet(DispatchKey::AutocastMTIA);
  constexpr auto autocast_maia_ks = DispatchKeySet(DispatchKey::AutocastMAIA);
  constexpr auto autocast_xpu_ks = DispatchKeySet(DispatchKey::AutocastXPU);
  constexpr auto autocast_ipu_ks = DispatchKeySet(DispatchKey::AutocastIPU);
  constexpr auto autocast_hpu_ks = DispatchKeySet(DispatchKey::AutocastHPU);
  constexpr auto autocast_cuda_ks = DispatchKeySet(DispatchKey::AutocastCUDA);
  constexpr auto autocast_xla_ks = DispatchKeySet(DispatchKey::AutocastXLA);
  constexpr auto autocast_privateuse1_ks =
      DispatchKeySet(DispatchKey::AutocastPrivateUse1);
  constexpr auto autocast_mps_ks = DispatchKeySet(DispatchKey::AutocastMPS);
  switch (t) {
    case BackendComponent::CPUBit:
      return autocast_cpu_ks;
    case BackendComponent::MTIABit:
      return autocast_mtia_ks;
    case BackendComponent::MAIABit:
      return autocast_maia_ks;
    case BackendComponent::XPUBit:
      return autocast_xpu_ks;
    case BackendComponent::IPUBit:
      return autocast_ipu_ks;
    case BackendComponent::HPUBit:
      return autocast_hpu_ks;
    case BackendComponent::CUDABit:
      return autocast_cuda_ks;
    case BackendComponent::XLABit:
      return autocast_xla_ks;
    case BackendComponent::PrivateUse1Bit:
      return autocast_privateuse1_ks;
    case BackendComponent::MPSBit:
      return autocast_mps_ks;
    default:
      return DispatchKeySet();
  }
}

inline DispatchKey highestPriorityBackendTypeId(DispatchKeySet ks) {
  return (ks & backend_functionality_keys).highestPriorityTypeId();
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias);

inline DispatchKey legacyExtractDispatchKey(DispatchKeySet s) {
  return (s - autograd_dispatch_keyset_with_ADInplaceOrView -
          autocast_dispatch_keyset -
          DispatchKeySet({DispatchKey::Functionalize,
                          DispatchKey::PythonTLSSnapshot,
                          DispatchKey::FuncTorchGradWrapper,
                          DispatchKey::FuncTorchVmapMode,
                          DispatchKey::FuncTorchBatched,
                          DispatchKey::Python}))
      .highestPriorityTypeId();
}

template <class T>
using is_not_DispatchKeySet = std::negation<std::is_same<DispatchKeySet, T>>;

// NOTE: remove_DispatchKeySet_arg_from_func is omitted because the
// c10::guts type-list utilities are not yet ported.  The template is
// only used by the PyTorch dispatcher internals, which are not part
// of this compatibility layer.

inline std::string toString(DispatchKeySet ts) {
  std::ostringstream oss;
  oss << ts;
  return oss.str();
}

inline std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  os << "DispatchKeySet(";
  bool first = true;
  for (auto k : ts) {
    if (!first) os << ", ";
    os << toString(k);
    first = false;
  }
  os << ")";
  return os;
}

inline bool isBackendDispatchKey(DispatchKey t) {
  return t >= DispatchKey::StartOfDenseBackends &&
         t <= DispatchKey::EndOfRuntimeBackendKeys;
}

inline DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  if (isPerBackendFunctionalityKey(t)) {
    DispatchKeySet result;
    for (uint8_t backend = 1; backend <= num_backends; ++backend) {
      result = result.add(toRuntimePerBackendFunctionalityKey(
          t, static_cast<BackendComponent>(backend)));
    }
    return result;
  }
  return DispatchKeySet(t);
}

inline bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k) {
  return getRuntimeDispatchKeySet(t).has(k);
}

inline DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
  if (t == DispatchKey::AutogradCPU) {
    return DispatchKeySet(DispatchKey::CPU);
  } else if (t == DispatchKey::AutogradCUDA) {
    return DispatchKeySet(DispatchKey::CUDA);
  } else if (t == DispatchKey::AutogradXPU) {
    return DispatchKeySet(DispatchKey::XPU);
  } else if (t == DispatchKey::AutogradIPU) {
    return DispatchKeySet(DispatchKey::IPU);
  } else if (t == DispatchKey::AutogradHPU) {
    return DispatchKeySet(DispatchKey::HPU);
  } else if (t == DispatchKey::AutogradLazy) {
    return DispatchKeySet(DispatchKey::Lazy);
  } else if (t == DispatchKey::AutogradMeta) {
    return DispatchKeySet(DispatchKey::Meta);
  } else if (t == DispatchKey::AutogradMPS) {
    return DispatchKeySet(DispatchKey::MPS);
  } else if (t == DispatchKey::AutogradPrivateUse1) {
    return DispatchKeySet(DispatchKey::PrivateUse1);
  } else if (t == DispatchKey::AutogradPrivateUse2) {
    return DispatchKeySet(DispatchKey::PrivateUse2);
  } else if (t == DispatchKey::AutogradPrivateUse3) {
    return DispatchKeySet(DispatchKey::PrivateUse3);
  } else if (t == DispatchKey::AutogradNestedTensor) {
    return DispatchKeySet(DispatchKey::NestedTensor);
  } else if (t == DispatchKey::AutogradOther) {
    return autogradother_backends;
  }
  return DispatchKeySet();
}

inline bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  if (alias == DispatchKey::Autograd) {
    return autograd_dispatch_keyset.has(k);
  } else if (alias == DispatchKey::CompositeImplicitAutograd) {
    return true;
  } else if (alias == DispatchKey::CompositeExplicitAutograd) {
    return k != DispatchKey::Autograd && !autograd_dispatch_keyset.has(k);
  }
  return false;
}

inline std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
  std::array<FunctionalityOffsetAndMask, num_functionality_keys> result{};
  uint16_t offset = 0;
  for (uint8_t i = 0; i < num_functionality_keys; ++i) {
    DispatchKey key = static_cast<DispatchKey>(i);
    if (isPerBackendFunctionalityKey(key)) {
      result[i] = FunctionalityOffsetAndMask(offset, full_backend_mask);
      offset += num_backends;
    } else {
      result[i] = FunctionalityOffsetAndMask(offset, 0);
      offset += 1;
    }
  }
  return result;
}

}  // namespace c10
