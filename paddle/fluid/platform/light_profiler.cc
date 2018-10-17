#include "paddle/fluid/platform/light_profiler.h"

namespace paddle {
namespace platform {

std::string LightTimer::DebugString() const {
  std::stringstream ss;
  ss << "Light Timer: \n";
  ss.setf(std::ios::left);
  int width = 10;
  int key_width = 35;
  ss << std::setw(key_width) << "key" << std::setw(width) << "total"
     << std::setw(width) << "average" << std::setw(width) << "max"
     << std::setw(width) << "max" << std::setw(width) << "count"
     << "\n";
  for (auto &r : records_) {
    ss << std::setw(key_width) << r.repr;
    ss << std::setw(width) << std::setprecision(3) << r.total;
    ss << std::setw(width) << std::setprecision(3) << r.average();
    ss << std::setw(width) << std::setprecision(3) << r.cell;
    ss << std::setw(width) << std::setprecision(3) << r.floor;
    ss << std::setw(width) << std::setprecision(3) << r.count;
    ss << "\n";
  }
  return ss.str();
}

}  // namespace platform
}  // namespace paddle
