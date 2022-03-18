from type_mapping import input_types_map, attr_types_map, output_type_map

def is_input(s):
    return s in input_types_map

def is_attr(s):
    return s in attr_types_map

def is_output(s):
    return s in output_type_map

def is_base_api(api):
    return "kernel" in api and "infer_meta" in api

def supports_selected_rows_kernel(api):
    return is_base_api(api) and len(api["kernel"]["func"]) == 2

def supports_inplace(api):
    return "inplace_map" in api
