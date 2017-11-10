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

#pragma once

#include <iostream>

#include "paddle/platform/variant.h"

namespace paddle {
namespace platform {

struct CPUPlace {
  // WORKAROUND: for some reason, omitting this constructor
  // causes errors with boost 1.59 and OSX
  CPUPlace() {}

  // needed for variant equality comparison
  inline bool operator==(const CPUPlace &) const { return true; }
  inline bool operator!=(const CPUPlace &) const { return false; }
};

struct GPUPlace {
  GPUPlace() : GPUPlace(0) {}
  explicit GPUPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  // needed for variant equality comparison
  inline bool operator==(const GPUPlace &o) const { return device == o.device; }
  inline bool operator!=(const GPUPlace &o) const { return !(*this == o); }

  int device;
};

struct IsGPUPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &) const { return false; }
  bool operator()(const GPUPlace &gpu) const { return true; }
};

// Define the max number of Place in bit length. i.e., the max number of places
// should be less equal than 2^(NUM_PLACE_TYPE_LIMIT_IN_BIT)
#define NUM_PLACE_TYPE_LIMIT_IN_BIT 4

typedef boost::variant<GPUPlace, CPUPlace> Place;

// static check number of place types is less equal than
// 2^(NUM_PLACE_TYPE_LIMIT_IN_BIT)
BOOST_MPL_ASSERT((boost::mpl::less_equal<
                  Place::types::size,
                  boost::mpl::long_<1 << NUM_PLACE_TYPE_LIMIT_IN_BIT>>));

void set_place(const Place &);
const Place &get_place();

const GPUPlace default_gpu();
const CPUPlace default_cpu();

bool is_gpu_place(const Place &);
bool is_cpu_place(const Place &);
bool places_are_same_class(const Place &, const Place &);

std::ostream &operator<<(std::ostream &, const Place &);

}  // namespace platform
}  // namespace paddle
