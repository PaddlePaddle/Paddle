cc_test(test_cclient SRCS test_cclient.c DEPS paddle_pserver_cclient)
add_style_check_target(test_cclient test_cclient.c)
