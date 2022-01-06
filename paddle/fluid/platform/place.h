/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

// #include <functional>
// #include <iostream>
// #include <vector>

#include "paddle/fluid/platform/enforce.h"
// #include "paddle/fluid/platform/variant.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/enforce_npu.h"
#endif

#include "paddle/pten/common/place.h"
namespace paddle {
namespace platform {

using Place = pten::Place;
using CPUPlace = pten::CPUPlace;
using CUDAPlace = pten::GPUPlace;
using CUDAPinnedPlace = pten::GPUPinnedPlace;
using NPUPlace = pten::NPUPlace;
using NPUPinnedPlace = pten::NPUPinnedPlace;
using XPUPlace = pten::XPUPlace;
using IPUPlace = pten::IPUPlace;
using MLUPlace = pten::MLUPlace;

// struct CPUPlace {
//   // WORKAROUND: for some reason, omitting this constructor
//   // causes errors with boost 1.59 and OSX
//   CPUPlace() {}

//   // needed for variant equality comparison
//   inline bool operator==(const CPUPlace &) const { return true; }
//   inline bool operator!=(const CPUPlace &) const { return false; }
//   inline bool operator<(const CPUPlace &) const { return false; }
// };

// struct CUDAPlace {
//   CUDAPlace() : CUDAPlace(0) {}
//   explicit CUDAPlace(int d) : device(d) {}

//   inline int GetDeviceId() const { return device; }
//   // needed for variant equality comparison
//   inline bool operator==(const CUDAPlace &o) const {
//     return device == o.device;
//   }
//   inline bool operator!=(const CUDAPlace &o) const { return !(*this == o); }
//   inline bool operator<(const CUDAPlace &o) const { return device < o.device; }

//   int device;
// };

// struct CUDAPinnedPlace {
//   CUDAPinnedPlace() {}

//   // needed for variant equality comparison
//   inline bool operator==(const CUDAPinnedPlace &) const { return true; }
//   inline bool operator!=(const CUDAPinnedPlace &) const { return false; }
//   inline bool operator<(const CUDAPinnedPlace &) const { return false; }
// };

// // Place for Baidu Kunlun Accelerator
// struct XPUPlace {
//   XPUPlace() : XPUPlace(0) {}
//   explicit XPUPlace(int d) : device(d) {}

//   inline int GetDeviceId() const { return device; }
//   // needed for variant equality comparison
//   inline bool operator==(const XPUPlace &o) const { return device == o.device; }
//   inline bool operator!=(const XPUPlace &o) const { return !(*this == o); }
//   inline bool operator<(const XPUPlace &o) const { return device < o.device; }

//   int device;
// };

// struct NPUPlace {
//   NPUPlace() : NPUPlace(0) {}
//   explicit NPUPlace(int d) : device(d) {}

//   inline int GetDeviceId() const { return device; }
//   // needed for variant equality comparison
//   inline bool operator==(const NPUPlace &o) const { return device == o.device; }
//   inline bool operator!=(const NPUPlace &o) const { return !(*this == o); }
//   inline bool operator<(const NPUPlace &o) const { return device < o.device; }

//   int device;
// };

// struct NPUPinnedPlace {
//   NPUPinnedPlace() {}

//   inline bool operator==(const NPUPinnedPlace &) const { return true; }
//   inline bool operator!=(const NPUPinnedPlace &) const { return false; }
//   inline bool operator<(const NPUPinnedPlace &) const { return false; }
// };
// struct IPUPlace {
//   IPUPlace() : IPUPlace(0) {}
//   explicit IPUPlace(int d) : device(d) {}

//   inline int GetDeviceId() const { return device; }
//   // needed for variant equality comparison
//   inline bool operator==(const IPUPlace &o) const { return device == o.device; }
//   inline bool operator!=(const IPUPlace &o) const { return !(*this == o); }
//   inline bool operator<(const IPUPlace &o) const { return device < o.device; }

//   int device;
// };

// struct MLUPlace {
//   MLUPlace() : MLUPlace(0) {}
//   explicit MLUPlace(int d) : device(d) {}

//   inline int GetDeviceId() const { return device; }
//   // needed for variant equality comparison
//   inline bool operator==(const MLUPlace &o) const { return device == o.device; }
//   inline bool operator!=(const MLUPlace &o) const { return !(*this == o); }
//   inline bool operator<(const MLUPlace &o) const { return device < o.device; }

//   int device;
// };

// struct PluggableDevicePlace {
//   PluggableDevicePlace() {}
//   explicit PluggableDevicePlace(const std::string &dev_type, int d)
//       : device_type(dev_type), device(d) {}
//   explicit PluggableDevicePlace(const std::string &dev_type)
//       : PluggableDevicePlace(dev_type, 0) {}
//   inline int GetDeviceId() const { return device; }
//   inline const std::string &GetDeviceType() const { return device_type; }

//   inline bool operator==(const PluggableDevicePlace &o) const {
//     return device_type == o.device_type && device == o.device;
//   }
//   inline bool operator!=(const PluggableDevicePlace &o) const {
//     return !(*this == o);
//   }
//   inline bool operator<(const PluggableDevicePlace &o) const {
//     return std::pair<std::string, int>(device_type, device) <
//            std::pair<std::string, int>(o.device_type, o.device);
//   }

//   std::string device_type;
//   int device;
// };

// struct IsCUDAPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return true; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsCPUPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return true; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsCUDAPinnedPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &cuda_pinned) const { return true; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsXPUPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return true; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsNPUPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return true; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsNPUPinnedPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return true; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsMLUPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return true; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };
// struct IsIPUPlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return true; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return false; }
// };

// struct IsPluggableDevicePlace : public boost::static_visitor<bool> {
//   bool operator()(const CPUPlace &) const { return false; }
//   bool operator()(const XPUPlace &) const { return false; }
//   bool operator()(const NPUPlace &) const { return false; }
//   bool operator()(const IPUPlace &) const { return false; }
//   bool operator()(const MLUPlace &) const { return false; }
//   bool operator()(const CUDAPlace &) const { return false; }
//   bool operator()(const CUDAPinnedPlace &) const { return false; }
//   bool operator()(const NPUPinnedPlace &) const { return false; }
//   bool operator()(const PluggableDevicePlace &) const { return true; }
// };

// class Place : public boost::variant<CUDAPlace, XPUPlace, NPUPlace, CPUPlace,
//                                     CUDAPinnedPlace, NPUPinnedPlace, IPUPlace,
//                                     MLUPlace, PluggableDevicePlace> {
//  private:
//   using PlaceBase =
//       boost::variant<CUDAPlace, XPUPlace, NPUPlace, CPUPlace, CUDAPinnedPlace,
//                      NPUPinnedPlace, IPUPlace, MLUPlace, PluggableDevicePlace>;

//  public:
//   Place() = default;
//   Place(const CPUPlace &cpu_place) : PlaceBase(cpu_place) {}     // NOLINT
//   Place(const XPUPlace &xpu_place) : PlaceBase(xpu_place) {}     // NOLINT
//   Place(const NPUPlace &npu_place) : PlaceBase(npu_place) {}     // NOLINT
//   Place(const MLUPlace &mlu_place) : PlaceBase(mlu_place) {}     // NOLINT
//   Place(const IPUPlace &ipu_place) : PlaceBase(ipu_place) {}     // NOLINT
//   Place(const CUDAPlace &cuda_place) : PlaceBase(cuda_place) {}  // NOLINT
//   Place(const CUDAPinnedPlace &cuda_pinned_place)                // NOLINT
//       : PlaceBase(cuda_pinned_place) {}
//   Place(const NPUPinnedPlace &npu_pinned_place)  // NOLINT
//       : PlaceBase(npu_pinned_place) {}
//   Place(const PluggableDevicePlace &pluggable_device_place)  // NOLINT
//       : PlaceBase(pluggable_device_place) {}

//   bool operator<(const Place &place) const {
//     return PlaceBase::operator<(static_cast<const PlaceBase &>(place));
//   }
//   bool operator==(const Place &place) const {
//     return PlaceBase::operator==(static_cast<const PlaceBase &>(place));
//   }
// };

using PlaceList = std::vector<Place>;

class PlaceHelper {
 public:
  static std::string GetDeviceType(const Place &place);
  static size_t GetDeviceId(const Place &place);
  static Place CreatePlace(const std::string &dev_type, size_t dev_id = 0);
};

bool is_gpu_place(const Place &);
bool is_xpu_place(const Place &);
bool is_npu_place(const Place &);
bool is_mlu_place(const Place &);
bool is_ipu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool is_npu_pinned_place(const Place &);
bool is_custom_place(const Place &p);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  switch (place.GetType()) {
    case pten::AllocationType::GPU: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::GPUPINNED: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::XPU: {
#ifdef PADDLE_WITH_XPU
      platform::XPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Paddle is not compiled with XPU. Cannot visit xpu device"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::NPU: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::NPUPINNED: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::IPU: {
#ifdef PADDLE_WITH_IPU
      platform::IPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with IPU. Cannot visit ipu device"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::MLU: {
#ifdef PADDLE_WITH_MLU
      platform::MLUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with MLU. Cannot visit mlu device"));
#endif
    }
    default: {
      platform::CPUPlace p;
      return visitor(p);
    }
  }

//   typename Visitor::result_type operator()(
//       const PluggableDevicePlace &pluggable_device) const {
// #ifdef PADDLE_WITH_PLUGGABLE_DEVICE
//     return visitor_(pluggable_device);
// #else
//     PADDLE_THROW(platform::errors::Unavailable(
//         "Paddle is not compiled with PluggableDevice. Cannot visit "
//         "PluggableDevice"));
//     return typename Visitor::result_type();
// #endif
//   }
// };

// template <typename Visitor>
// typename Visitor::result_type VisitPlace(const Place &place,
//                                          const Visitor &visitor) {
//   return boost::apply_visitor(PlaceVisitorWrapper<Visitor>(visitor), place);
}

}  // namespace platform
}  // namespace paddle
