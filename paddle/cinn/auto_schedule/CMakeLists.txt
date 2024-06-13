add_subdirectory(analysis)
add_subdirectory(search_space)

cinn_proto_library(auto_schedule_proto SRCS auto_schedule.proto DEPS
                   schedule_desc_proto)
