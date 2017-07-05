# FIXME:It's ugly
#cc_test(test_cclient SRCS test_cclient.c DEPS paddle_pserver_cclient paddle_go_optimizer)
cc_test(test_cclient SRCS test_cclient.c DEPS paddle_pserver_cclient paddle_optimizer paddle_proto glog gflags protobuf)
add_style_check_target(test_cclient test_cclient.c)
