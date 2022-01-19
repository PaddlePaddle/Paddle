proto_library(nodetreeproto SRCS nodetree.proto)
cc_library(serialization_logger SRCS serialization_logger.cc DEPS nodetreeproto)
cc_library(deserialization_reader SRCS deserialization_reader.cc DEPS nodetreeproto)
