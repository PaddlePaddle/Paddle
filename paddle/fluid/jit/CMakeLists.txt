proto_library(paddle_jit_property_proto SRCS property.proto)

file(GLOB_RECURSE fluid_jit_srcs "*.cc")
set(fluid_jit_deps paddle_jit_property_proto proto_desc executor
                   compiled_program)

cc_library(
  fluid_jit
  SRCS ${fluid_jit_srcs}
  DEPS ${fluid_jit_deps})

add_dependencies(fluid_jit xxhash)
