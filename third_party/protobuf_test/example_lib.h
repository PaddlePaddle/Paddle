#pragma once

#include "third_party/protobuf_test/example.pb.h"

#include <string>

std::string get_greet(const ::protos::Greeting &who);
