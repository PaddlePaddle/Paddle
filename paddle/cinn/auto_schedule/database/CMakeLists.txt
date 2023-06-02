core_gather_headers()

gather_srcs(cinnapi_src SRCS database.cc jsonfile_database.cc)

cinn_cc_test(test_database SRCS database_test.cc DEPS cinncore)
cinn_cc_test(test_jsonfile_database SRCS jsonfile_database_test.cc DEPS
             cinncore)
