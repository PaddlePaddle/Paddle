# External dependency to Google protobuf.
http_archive(
    name="protobuf",
    url="http://github.com/google/protobuf/archive/v3.1.0.tar.gz",
    sha256="0a0ae63cbffc274efb573bdde9a253e3f32e458c41261df51c5dbc5ad541e8f7",
    strip_prefix="protobuf-3.1.0", )

# External dependency to gtest 1.7.0.  This method comes from
# https://www.bazel.io/versions/master/docs/tutorial/cpp.html.
new_http_archive(
    name="gtest",
    url="https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256="b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file="third_party/gtest.BUILD",
    strip_prefix="googletest-release-1.7.0", )
