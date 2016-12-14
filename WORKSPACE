# External dependency to Google protobuf.
http_archive(
    name="protobuf",
    url="http://github.com/google/protobuf/archive/v3.1.0.tar.gz",
    sha256="0a0ae63cbffc274efb573bdde9a253e3f32e458c41261df51c5dbc5ad541e8f7",
    strip_prefix="protobuf-3.1.0")

# External dependency to gtest 1.7.0.  This method comes from
# https://www.bazel.io/versions/master/docs/tutorial/cpp.html.
new_http_archive(
    name="gtest",
    url="https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256="b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file="third_party/gtest.BUILD",
    strip_prefix="googletest-release-1.7.0")

# External dependency to gflags.  This method comes from
# https://github.com/gflags/example/blob/master/WORKSPACE.
new_git_repository(
    name="gflags",
    tag="v2.2.0",
    remote="https://github.com/gflags/gflags.git",
    build_file="third_party/gflags.BUILD")

# External dependency to glog.  This method comes from
# https://github.com/reyoung/bazel_playground/blob/master/WORKSPACE
new_git_repository(
    name="glog",
    remote="https://github.com/google/glog.git",
    commit="b6a5e0524c28178985f0d228e9eaa43808dbec3c",
    build_file="third_party/glog.BUILD")
