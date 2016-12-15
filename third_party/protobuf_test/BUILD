licenses(["notice"])  # Apache 2.0

load("@protobuf//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name="example_proto",
    srcs=["example.proto"],
    protoc="@protobuf//:protoc",
    default_runtime="@protobuf//:protobuf", )

cc_library(
    name="example_lib",
    srcs=["example_lib.cc"],
    hdrs=["example_lib.h"],
    deps=[":example_proto"], )

cc_test(
    name="example_lib_test",
    srcs=["example_lib_test.cc"],
    copts=["-Iexternal/gtest/include"],
    deps=[
        "@gtest//:gtest",
        ":example_lib",
    ], )
