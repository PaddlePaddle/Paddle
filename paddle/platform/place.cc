/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/platform/place.h"

namespace paddle {
namespace platform {

namespace detail {

class PlacePrinter : public boost::static_visitor<> {
 public:
  explicit PlacePrinter(std::ostream &os) : os_(os) {}
  void operator()(const CPUPlace &) { os_ << "CPUPlace"; }
  void operator()(const GPUPlace &p) { os_ << "GPUPlace(" << p.device << ")"; }

 private:
  std::ostream &os_;
};

}  // namespace detail

static Place the_default_place;

void set_place(const Place &place) { the_default_place = place; }
const Place &get_place() { return the_default_place; }

const GPUPlace default_gpu() { return GPUPlace(0); }
const CPUPlace default_cpu() { return CPUPlace(); }

bool is_gpu_place(const Place &p) {
  return boost::apply_visitor(IsGPUPlace(), p);
}
bool is_cpu_place(const Place &p) {
  return !boost::apply_visitor(IsGPUPlace(), p);
}

bool places_are_same_class(const Place &p1, const Place &p2) {
  return p1.which() == p2.which();
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  detail::PlacePrinter printer(os);
  boost::apply_visitor(printer, p);
  return os;
}

}  // namespace platform
}  // namespace paddle
