#include <gflags/gflags.h>
#include <iostream>
#include <sstream>
#include "paddle/fluid/string/printf.h"

DEFINE_bool(color, true, "Whether to turn on pretty log");

namespace paddle {
namespace framework {
namespace analysis {

namespace logging {

inline std::string black() { return FLAGS_color ? "\e[30m" : ""; }
inline std::string red() { return FLAGS_color ? "\e[31m" : ""; }
inline std::string b_red() { return FLAGS_color ? "\e[41m" : ""; }
inline std::string green() { return FLAGS_color ? "\e[32m" : ""; }
inline std::string yellow() { return FLAGS_color ? "\e[33m" : ""; }
inline std::string blue() { return FLAGS_color ? "\e[34m" : ""; }
inline std::string purple() { return FLAGS_color ? "\e[35m" : ""; }
inline std::string cyan() { return FLAGS_color ? "\e[36m" : ""; }
inline std::string light_gray() { return FLAGS_color ? "\e[37m" : ""; }
inline std::string white() { return FLAGS_color ? "\e[37m" : ""; }
inline std::string light_red() { return FLAGS_color ? "\e[91m" : ""; }
inline std::string dim() { return FLAGS_color ? "\e[2m" : ""; }
inline std::string bold() { return FLAGS_color ? "\e[1m" : ""; }
inline std::string underline() { return FLAGS_color ? "\e[4m" : ""; }
inline std::string blink() { return FLAGS_color ? "\e[5m" : ""; }
inline std::string reset() { return FLAGS_color ? "\e[0m" : ""; }

using TextBlock = std::pair<std::string, std::string>;

inline std::string info() { return black(); }
inline std::string warn() { return b_red(); }
inline std::string suc() { return green(); }

static void PrettyLog(const std::vector<TextBlock>& texts) {
  std::stringstream ss;
  for (const auto& t : texts) {
    ss << t.first << t.second << reset();
  }
  std::cerr << ss.str() << std::endl;
}

template <typename... Args>
static void PrettyLog(const std::string& style, const char* fmt,
                      const Args&... args, bool end = true) {
  std::cerr << style << strings::Sprintf(fmt, args...)
            << (end ? std::endl : "");
}

}  // namespace logging
}  // namespace analysis
}  // namespace framework

}  // namespace paddle
