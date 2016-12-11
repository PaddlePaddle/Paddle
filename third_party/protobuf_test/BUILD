licenses(["notice"])  # Apache 2.0

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_library")

cpp_proto_library(
    name = "example_proto",
    protos = [
        "example.proto"
    ],
)

cc_library(
    name = "example_lib",
    srcs = ["example_lib.cc"],
    hdrs = ["example_lib.h"],
    deps = [":example_proto"],
)
