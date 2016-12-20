# Bazel (http://bazel.io/) BUILD file for gflags.
#
# See INSTALL.md for instructions for adding gflags to a Bazel workspace.

licenses(["notice"])

exports_files(["src/gflags_complections.sh", "COPYING.txt"])

load(":bazel/gflags.bzl", "gflags_sources", "gflags_library")
(hdrs, srcs) = gflags_sources(namespace=["google", "gflags"])
gflags_library(hdrs=hdrs, srcs=srcs, threads=0)
gflags_library(hdrs=hdrs, srcs=srcs, threads=1)
