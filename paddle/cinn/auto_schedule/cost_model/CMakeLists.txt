core_gather_headers()

gather_srcs(cinnapi_src SRCS xgb_cost_model.cc expr_cost_model.cc feature.cc
            feature_extractor.cc)

# TODO(zhhsplendid): enable this test again
#cinn_cc_test(test_xgb_cost_model SRCS xgb_cost_model_test.cc DEPS cinncore)
cinn_cc_test(test_feature_extractor SRCS feature_extractor_test.cc DEPS
             cinncore)
cinn_cc_test(test_feature SRCS feature_test.cc DEPS cinncore)
