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