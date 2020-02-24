cc_library(op_desc_meta SRCS op_desc_meta.cc DEPS proto_desc layer)
cc_library(program_desc_tracer SRCS program_desc_tracer.cc DEPS op_desc_meta)
