cc_binary(execute_program_desc SRCS execute_program_desc.cc DEPS proto_desc memory executor prune init
    profiler feed_fetch_method parallel_executor ${GLOB_OP_LIB}) 
