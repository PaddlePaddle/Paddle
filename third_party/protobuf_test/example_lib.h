#pragma once

#include "third_party/protobuf_test/example.pb.h"

#include <string>

namespace third_party {
namespace protobuf_test {

std::string get_greet(const Greeting &who);

}  // namespace protobuf_test
}  // namespace third_party
