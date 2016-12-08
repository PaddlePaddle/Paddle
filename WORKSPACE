http_archive(
    name = "protobuf",
    url = "http://github.com/google/protobuf/archive/008b5a228b37c054f46ba478ccafa5e855cb16db.tar.gz",
    sha256 = "2737ad055eb8a9bc63ed068e32c4ea280b62d8236578cb4d4120eb5543f759ab",
    strip_prefix = "protobuf-008b5a228b37c054f46ba478ccafa5e855cb16db",
)

bind(
    name = "protobuf_clib",
    actual = "@protobuf//:protoc_lib",
)


bind(
    name = "protobuf_compiler",
    actual = "@protobuf//:protoc_lib",
)

new_http_archive(
    name = "six_archive",
    url = "http://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "third_party/six.BUILD",
)
bind(
    name = "six",
    actual = "@six_archive//:six",
)


new_http_archive(
    name = "python_archive",
    url="http://paddlepaddle.bj.bcebos.com/bazel/python/macos/python27.mac.bin.tar.gz",
    sha256="4905cf20272b8118f4f0afb570549ee0b492280f7eb12a51bf79dd79f2e871fe",
    build_file = "third_party/python.BUILD",
)

bind(
    name = "libpython",
    actual = "@python_archive//:libpython"
)