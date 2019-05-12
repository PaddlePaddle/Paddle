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

#include "paddle/fluid/platform/place.h"

DEFINE_bool(benchmark, false,
            "Doing memory benchmark. It will make deleting scope synchronized, "
            "and add some memory usage logs."
            "Default cuda is asynchronous device, set to True will"
            "force op run in synchronous mode.");

namespace paddle {
namespace platform {

namespace detail {

class PlacePrinter : public boost::static_visitor<> {
 public:
  explicit PlacePrinter(std::ostream &os) : os_(os) {}
  void operator()(const CPUPlace &) { os_ << "CPUPlace"; }
  void operator()(const CUDAPlace &p) {
    os_ << "CUDAPlace(" << p.device << ")";
  }
  void operator()(const CUDAPinnedPlace &p) { os_ << "CUDAPinnedPlace"; }

 private:
  std::ostream &os_;
};

}  // namespace detail

static Place the_default_place;

void set_place(const Place &place) { the_default_place = place; }
const Place &get_place() { return the_default_place; }

const CUDAPlace default_gpu() { return CUDAPlace(0); }
const CPUPlace default_cpu() { return CPUPlace(); }
const CUDAPinnedPlace default_cuda_pinned() { return CUDAPinnedPlace(); }

bool is_gpu_place(const Place &p) {
  return boost::apply_visitor(IsCUDAPlace(), p);
}

bool is_cpu_place(const Place &p) {
  return boost::apply_visitor(IsCPUPlace(), p);
}

bool is_cuda_pinned_place(const Place &p) {
  return boost::apply_visitor(IsCUDAPinnedPlace(), p);
}

bool places_are_same_class(const Place &p1, const Place &p2) {
  return p1.which() == p2.which();
}

bool is_same_place(const Place &p1, const Place &p2) {
  if (places_are_same_class(p1, p2)) {
    if (is_cpu_place(p1) || is_cuda_pinned_place(p1)) {
      return true;
    } else {
      return boost::get<CUDAPlace>(p1) == boost::get<CUDAPlace>(p2);
    }
  } else {
    return false;
  }
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  detail::PlacePrinter printer(os);
  boost::apply_visitor(printer, p);
  return os;
}

}  // namespace platform
}  // namespace paddle
