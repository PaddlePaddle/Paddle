# type mapping: types in yaml -> types in c++ API
input_types_map = {
    'Tensor': 'const Tensor&',
    'Tensor[]': 'const std::vector<Tensor>&'
}

optional_input_types_map = {
    'Tensor': 'const paddle::optional<Tensor>&',
    'Tensor[]': 'const paddle::optional<std::vector<Tensor>>&',
}

attr_types_map = {
    # special types
    'ScalarArray': 'const ScalarArray&',
    'Scalar': 'const Scalar&',
    'Backend': 'Backend',
    'DataLayout': 'DataLayout',
    'DataType': 'DataType',
    # scalar types
    'bool': 'bool',
    'int': 'int',
    'int64': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'const std::string&',
    # vector types
    'bool[]': 'const std::vector<bool>&',
    'int[]': 'const std::vector<int>&',
    'int64[]': 'const std::vector<int64_t>&',
    'float[]': 'const std::vector<float>&',
    'double[]': 'const std::vector<double>&',
    'str[]': 'const std::vector<<std::string>&',
}

opmaker_attr_types_map = {
    # special types
    'ScalarArray': 'std::vector<int64_t>',
    'Scalar': 'double',
    'Backend': 'int',
    'DataLayout': 'int',
    'DataType': 'int',
    # scalar types
    'bool': 'bool',
    'int': 'int',
    'int64': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'std::string',
    # vector types
    'bool[]': 'std::vector<bool>',
    'int[]': 'std::vector<int>',
    'int64[]': 'std::vector<int64_t>',
    'float[]': 'std::vector<float>',
    'double[]': 'std::vector<double>',
    'str[]': 'std::vector<<std::string>',
}

output_type_map = {'Tensor': 'Tensor', 'Tensor[]': 'std::vector<Tensor>'}

#------------------------------ phi attr ------------------------------
phi_attr_types_map = attr_types_map.copy()
phi_attr_types_map.update({
    'ScalarArray': 'const phi::ScalarArray&',
    'Scalar': 'const phi::Scalar&'
})

#--------------------------- phi dense tensor ---------------------------
# type mapping to phi, used in implementation
dense_input_types_map = {
    'Tensor': 'const phi::DenseTensor&',
    'Tensor[]': 'const std::vector<const phi::DenseTensor*>&',
}

dense_optional_input_types_map = {
    'Tensor': 'paddle::optional<const phi::DenseTensor&>',
    'Tensor[]': 'paddle::optional<const std::vector<phi::DenseTensor>&>'
}

dense_output_types_map = {
    'Tensor': 'phi::DenseTensor*',
    'Tensor[]': 'std::vector<phi::DenseTensor*>&'
}

#---------------------- phi selected rows------------------------------
# type mapping to phi, used in implementation
sr_input_types_map = {
    'Tensor': 'const phi::SelectedRows&',
}

sr_optional_input_types_map = {
    'Tensor': 'paddle::optional<const phi::SelectedRows&>',
}

sr_output_types_map = {
    'Tensor': 'phi::SelectedRows*',
}
