
cc_binary(main SRCS main.c DEPS paddle_pserver_cclient)
cc_test(test_cclient SRCS test_cclient.c DEPS paddle_pserver_cclient)
