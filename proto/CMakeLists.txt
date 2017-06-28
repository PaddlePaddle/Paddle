file(GLOB proto_filenames . *.proto)
pb_cc_library(paddle_proto SRCS ${proto_filenames})
pb_py_library(gen_proto_py SRCS ${proto_filenames} TARGET_DIR ${PROJ_ROOT}/python/paddle/proto)
