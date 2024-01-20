GLOG_v=4 FLAGS_cinn_convert_static_dim_to_dynamic=2048:S0 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True python test_rms_norm.py 2>&1 | tee /tmp/a
# GLOG_v=4 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True python test_rms_norm.py 2>&1 | tee /tmp/a
