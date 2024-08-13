core_gather_headers()

gather_srcs(cinnapi_src SRCS base_group_scheduler.cc)
gather_srcs(cinnapi_src SRCS dy_shape_group_scheduler.cc)

add_subdirectory(config)
add_subdirectory(tactic)
add_subdirectory(search)
