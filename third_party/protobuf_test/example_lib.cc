#include "third_party/protobuf_test/example_lib.h"
#include <string>

std::string get_greet(const ::protos::Greeting& who) {
  return "Hello " + who.name();
}
