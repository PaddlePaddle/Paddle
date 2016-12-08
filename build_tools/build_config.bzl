load("@protobuf//:protobuf.bzl", "cc_proto_library")
load("@protobuf//:protobuf.bzl", "py_proto_library")

# TODO(): How to get configuration from command line or environment variable
USE_DOUBLE = False

# TODO(): Parse PADDLE_VERSION from git tag.
PADDLE_VERSION = "0.9.0"

if USE_DOUBLE:
    ACCURACY = "double"
else:
    ACCURACY = "float"


def pd_proto_library(name, srcs=[]):
    generated_proto_names = []
    for each_proto in srcs:
        proto_name = 'generate_proto_files_' + each_proto
        generated_proto_names.append(proto_name)
        native.genrule(
            name=proto_name,
            srcs=[each_proto + ".proto.m4"],
            outs=[each_proto + ".proto"],
            cmd="m4 -Dreal=" + ACCURACY + " -Dproto3 $< > $@")

    cc_proto_library(
        name=name + "_cc",
        srcs=[s + ".proto" for s in srcs],
        cc_libs=["@protobuf//:protobuf"],
        protoc="@protobuf//:protoc",
        default_runtime="@protobuf//:protobuf",
        visibility=["//visibility:public"],
        include=".")

    py_proto_library(
        name=name + "_py",
        srcs=[s + ".proto" for s in srcs],
        deps=["@protobuf//:protobuf_python"],
        protoc="@protobuf//:protoc",
        default_runtime="@protobuf//:protobuf_python",
        visibility=["//visibility:public"],
        include=".")
