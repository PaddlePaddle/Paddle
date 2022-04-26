
cc_library(afs_wrapper SRCS afs_warpper.cc DEPS fs ps_framework_proto)
add_dependencies(afs_wrapper ps.pb.h ps.pb.cc)

#set_property(GLOBAL PROPERTY COMMON_DEPS afs_warpper)
