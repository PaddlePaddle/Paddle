#include <majel/place.h>
#include <mapbox/variant.hpp>

namespace majel {

namespace detail {

class PlacePrinter : public mapbox::util::static_visitor<> {
private:
  std::ostream& os_;

public:
  PlacePrinter(std::ostream& os) : os_(os) {}

  void operator()(const CpuPlace&) { os_ << "CpuPlace"; }

  void operator()(const GpuPlace& p) { os_ << "GpuPlace(" << p.device << ")"; }
};

}  // namespace majel

static Place the_default_place;

void set_place(const Place& place) { the_default_place = place; }

const Place& get_place() { return the_default_place; }

const GpuPlace default_gpu() { return GpuPlace(0); }

const CpuPlace default_cpu() { return CpuPlace(); }

bool is_gpu_place(const Place& p) {
  return mapbox::util::apply_visitor(IsGpuPlace(), p);
}

bool is_cpu_place(const Place& p) {
  return !mapbox::util::apply_visitor(IsGpuPlace(), p);
}

bool places_are_same_class(const Place& p1, const Place& p2) {
  return is_gpu_place(p1) == is_gpu_place(p2);
}

std::ostream& operator<<(std::ostream& os, const majel::Place& p) {
  majel::detail::PlacePrinter printer(os);
  mapbox::util::apply_visitor(printer, p);
  return os;
}

}  // namespace majel
