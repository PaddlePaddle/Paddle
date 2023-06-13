py_test_modules(test_standalone_new_ir MODULES test_standalone_executor ENVS
                GLOG_v=1 FLAGS_enable_new_ir_in_executor=true)
