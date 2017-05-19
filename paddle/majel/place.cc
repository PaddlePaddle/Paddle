#include "place.h"

namespace majel {

namespace detail {

class PlacePrinter : public boost::static_visitor<> {
private:
  std::ostream& os_;

public:
  PlacePrinter(std::ostream& os) : os_(os) {}

  void operator()(const CpuPlace&) { os_ << "CpuPlace"; }

  void operator()(const GpuPlace&) { os_ << "GpuPlace"; }
};

}  // namespace detail

static Place the_default_place;

const Place& get_place() { return the_default_place; }
bool is_gpu_place(const Place& p) {
  return boost::apply_visitor(IsGpuPlace(), p);
}

bool is_cpu_place(const Place& p) {
  return !boost::apply_visitor(IsGpuPlace(), p);
}

bool places_are_same_class(const Place& p1, const Place& p2) {
  return is_gpu_place(p1) == is_gpu_place(p2);
}

std::ostream& operator<<(std::ostream& os, const majel::Place& p) {
  majel::detail::PlacePrinter printer(os);
  boost::apply_visitor(printer, p);
  return os;
}

}  // namespace majel
