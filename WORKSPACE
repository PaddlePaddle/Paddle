# External dependency to grpc-enabled Google-styleprotobuf bulding
# rules.  This method comes from
# https://github.com/pubref/rules_protobuf#usage.
git_repository(
    name = "org_pubref_rules_protobuf",
    remote = "https://github.com/pubref/rules_protobuf",
    tag = "v0.7.1",
)

# External dependency to gtest 1.7.0.  This method comes from
# https://www.bazel.io/versions/master/docs/tutorial/cpp.html.
new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "third_party/gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)


load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")
cpp_proto_repositories()
