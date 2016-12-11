licenses(["notice"])  # Apache 2.0

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_library")

cpp_proto_library(
    name = "example_proto",
    protos = [
        "example.proto"
    ],
    with_grpc = True,
)

cc_library(
    name = "example_lib",
    srcs = ["example_lib.cc"],
    hdrs = ["example_lib.h"],
    deps = [":example_proto"],
)

cc_test(
    name = "example_lib_test",
    srcs = ["example_lib_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps =[
        "@gtest//:main",
        ":example_lib",
    ],
)
