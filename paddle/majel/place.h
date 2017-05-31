#pragma once
#include <boost/variant.hpp>
#include <iostream>

namespace majel {

struct CpuPlace {
  CpuPlace() {}  // WORKAROUND: for some reason, omitting this constructor
                 // causes errors with boost 1.59 and OSX
  // needed for variant equality comparison
  inline bool operator==(const CpuPlace&) const { return true; }

  inline bool operator!=(const CpuPlace&) const { return false; }
};

struct GpuPlace {
  GpuPlace() {}

  // needed for variant equality comparison
  inline bool operator==(const GpuPlace&) const { return true; }

  inline bool operator!=(const GpuPlace&) const { return false; }
};

class IsGpuPlace : public boost::static_visitor<bool> {
public:
  bool operator()(const CpuPlace&) const { return false; }

  bool operator()(const GpuPlace&) const { return true; }
};

typedef boost::variant<CpuPlace, GpuPlace> Place;

const Place& get_place();
bool is_gpu_place(const Place&);
bool is_cpu_place(const Place&);
bool places_are_same_class(const Place&, const Place&);

std::ostream& operator<<(std::ostream&, const majel::Place&);

}  // namespace majel
