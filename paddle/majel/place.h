#pragma once
#include <iostream>
#include "mapbox/variant.hpp"

namespace majel {

struct CpuPlace {
  CpuPlace() {}  // WORKAROUND: for some reason, omitting this constructor
                 // causes errors with boost 1.59 and OSX
  // needed for variant equality comparison
  inline bool operator==(const CpuPlace&) const { return true; }

  inline bool operator!=(const CpuPlace&) const { return false; }
};

struct GpuPlace {
  GpuPlace(int d) : device(d) {}

  // needed for variant equality comparison
  inline bool operator==(const GpuPlace& o) const { return device == o.device; }

  inline bool operator!=(const GpuPlace& o) const { return !(*this == o); }

  GpuPlace() : GpuPlace(0) {}
  int device;
};

class IsGpuPlace : public mapbox::util::static_visitor<bool> {
public:
  bool operator()(const CpuPlace&) const { return false; }

  bool operator()(const GpuPlace& gpu) const { return true; }
};

typedef mapbox::util::variant<GpuPlace, CpuPlace> Place;

void set_place(const Place&);

const Place& get_place();

const GpuPlace default_gpu();
const CpuPlace default_cpu();

bool is_gpu_place(const Place&);
bool is_cpu_place(const Place&);
bool places_are_same_class(const Place&, const Place&);

std::ostream& operator<<(std::ostream&, const majel::Place&);

}  // namespace majel
