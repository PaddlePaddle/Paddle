op_name_mappings = [
    { "adadelta": "adadelta_" },
    { "adagrad": "adagrad_" },
    { "adam": "adam_" },
    { "adamax": "adamax_" },
    { "adamw": "adamw_" },
    { "elementwise_add": "add" },
    { "elementwise_add_grad": "add_grad" },
    { "elementwise_add_grad_grad": "add_double_grad" },
    { "elementwise_add_triple_grad": "add_triple_grad" },
    { "sum": "add_n" },
    { "reduce_all": "all" },
    { "reduce_amax": "amax" },
    { "reduce_amax_grad": "amax_grad" },
    { "reduce_amin": "amin" },
    { "reduce_amin_grad": "amin_grad" },
    { "reduce_any": "any" },
    { "range": "arange" },
    { "arg_max": "argmax" },
    { "arg_min": "argmin" },
    { "tensor_array_to_tensor": "array_to_tensor" },
    { "batch_norm_grad_grad": "batch_norm_double_grad" },
    { "bicubic_interp_v2": "bicubic_interp" },
    { "bicubic_interp_v2_grad": "bicubic_interp_grad" },
    { "bilinear_tensor_product": "bilinear" },
    { "bilinear_tensor_product_grad": "bilinear_grad" },
    { "bilinear_interp_v2": "bilinear_interp" },
    { "bilinear_interp_v2_grad": "bilinear_interp_grad" },
    { "celu_grad_grad": "celu_double_grad" },
    { "check_finite_and_unscale": "check_finite_and_unscale_" },
    { "conv2d_transpose_grad_grad": "conv2d_transpose_double_grad" },
    { "conv3d_grad_grad": "conv3d_double_grad" },
    { "crop_tensor": "crop" },
    { "crop_tensor_grad": "crop_grad" },
    { "softmax_with_cross_entropy": "cross_entropy_with_softmax" },
    { "softmax_with_cross_entropy_grad": "cross_entropy_with_softmax_grad" },
    { "depthwise_conv2d_grad_grad": "depthwise_conv2d_double_grad" },
    { "determinant": "det" },
    { "determinant_grad": "det_grad" },
    { "diag_v2": "diag" },
    { "diag_v2_grad": "diag_grad" },
    { "elementwise_div": "divide" },
    { "elementwise_div_grad": "divide_grad" },
    { "elu_grad_grad": "elu_double_grad" },
    { "lookup_table_v2": "embedding" },
    { "lookup_table_v2_grad": "embedding_grad" },
    { "expand_v2": "expand" },
    { "expand_v2_grad": "expand_grad" },
    { "expand_v2_double_grad": "expand_double_grad" },
    { "expand_as_v2": "expand_as" },
    { "expand_as_v2_grad": "expand_as_grad" },
    { "exponential": "exponential_" },
    { "exponential_grad": "exponential__grad" },
    { "fill_any": "fill" },
    { "fill_any_grad": "fill_grad" },
    { "flatten_contiguous_range": "flatten" },
    { "flatten_contiguous_range_grad": "flatten_grad" },
    { "elementwise_floordiv": "floor_divide" },
    { "elementwise_fmax": "fmax" },
    { "elementwise_fmax_grad": "fmax_grad" },
    { "elementwise_fmin": "fmin" },
    { "elementwise_fmin_grad": "fmin_grad" },
    { "fill_constant": "full" },
    { "fill_any_like": "full_like" },
    { "fused_bn_add_activation": "fused_bn_add_activation_" },
    { "gaussian_random": "gaussian" },
    { "generate_proposals_v2": "generate_proposals" },
    { "grid_sampler": "grid_sample" },
    { "grid_sampler_grad": "grid_sample_grad" },
    { "hard_shrink": "hardshrink" },
    { "hard_shrink_grad": "hardshrink_grad" },
    { "hard_sigmoid": "hardsigmoid" },
    { "hard_sigmoid_grad": "hardsigmoid_grad" },
    { "hard_swish": "hardswish" },
    { "hard_swish_grad": "hardswish_grad" },
    { "brelu": "hardtanh" },
    { "brelu_grad": "hardtanh_grad" },
    { "elementwise_heaviside": "heaviside" },
    { "elementwise_heaviside_grad": "heaviside_grad" },
    { "hierarchical_sigmoid": "hsigmoid_loss" },
    { "hierarchical_sigmoid_grad": "hsigmoid_loss_grad" },
    { "isfinite_v2": "isfinite" },
    { "isinf_v2": "isinf" },
    { "isnan_v2": "isnan" },
    { "lamb": "lamb_" },
    { "leaky_relu_grad_grad": "leaky_relu_double_grad" },
    { "linear_interp_v2": "linear_interp" },
    { "linear_interp_v2_grad": "linear_interp_grad" },
    { "log_grad_grad": "log_double_grad" },
    { "matmul_v2": "matmul" },
    { "matmul_v2_grad": "matmul_grad" },
    { "matmul_v2_grad_grad": "matmul_double_grad" },
    { "matmul_v2_triple_grad": "matmul_triple_grad" },
    { "mul": "matmul_with_flatten" },
    { "mul_grad": "matmul_with_flatten_grad" },
    { "reduce_max": "max" },
    { "reduce_max_grad": "max_grad" },
    { "elementwise_max": "maximum" },
    { "elementwise_max_grad": "maximum_grad" },
    { "reduce_mean": "mean" },
    { "reduce_mean_grad": "mean_grad" },
    { "mean": "mean_all" },
    { "mean_grad": "mean_all_grad" },
    { "merged_momentum": "merged_momentum_" },
    { "reduce_min": "min" },
    { "reduce_min_grad": "min_grad" },
    { "elementwise_min": "minimum" },
    { "elementwise_min_grad": "minimum_grad" },
    { "momentum": "momentum_" },
    { "elementwise_mul": "multiply" },
    { "elementwise_mul_grad": "multiply_grad" },
    { "nearest_interp_v2": "nearest_interp" },
    { "nearest_interp_v2_grad": "nearest_interp_grad" },
    { "where_index": "nonzero" },
    { "size": "numel" },
    { "one_hot_v2": "one_hot" },
    { "reduce_prod": "prod" },
    { "reduce_prod_grad": "prod_grad" },
    { "relu_grad_grad": "relu_double_grad" },
    { "elementwise_mod": "remainder" },
    { "reshape2": "reshape" },
    { "reshape2_grad": "reshape_grad" },
    { "rmsprop": "rmsprop_" },
    { "rsqrt_grad_grad": "rsqrt_double_grad" },
    { "graph_send_recv": "send_u_recv" },
    { "graph_send_recv_grad": "send_u_recv_grad" },
    { "graph_send_ue_recv": "send_ue_recv" },
    { "graph_send_ue_recv_grad": "send_ue_recv_grad" },
    { "graph_send_uv": "send_uv" },
    { "graph_send_uv_grad": "send_uv_grad" },
    { "sgd": "sgd_" },
    { "sigmoid_grad_grad": "sigmoid_double_grad" },
    { "slogdeterminant": "slogdet" },
    { "slogdeterminant_grad": "slogdet_grad" },
    { "sqrt_grad_grad": "sqrt_double_grad" },
    { "square_grad_grad": "square_double_grad" },
    { "squeeze2": "squeeze" },
    { "squeeze2_grad": "squeeze_grad" },
    { "squeeze2_double_grad": "squeeze_double_grad" },
    { "elementwise_sub": "subtract" },
    { "elementwise_sub_grad": "subtract_grad" },
    { "reduce_sum": "sum" },
    { "reduce_sum_grad": "sum_grad" },
    { "tanh_grad_grad": "tanh_double_grad" },
    { "top_k_v2": "topk" },
    { "top_k_v2_grad": "topk_grad" },
    { "transpose2": "transpose" },
    { "transpose2_grad": "transpose_grad" },
    { "trilinear_interp_v2": "trilinear_interp" },
    { "trilinear_interp_v2_grad": "trilinear_interp_grad" },
    { "uniform_random": "uniform" },
    { "uniform_random_inplace": "uniform_inplace" },
    { "uniform_random_inplace_grad": "uniform_inplace_grad" },
    { "unsqueeze2": "unsqueeze" },
    { "unsqueeze2_grad": "unsqueeze_grad" },
    { "unsqueeze2_double_grad": "unsqueeze_double_grad" },
    { "update_loss_scaling": "update_loss_scaling_" },
    { "yolov3_loss": "yolo_loss" },
    { "yolov3_loss_grad": "yolo_loss_grad" },
    { "fetch_v2": "fetch" },
    { "fill_constant_batch_size_like": "full_batch_size_like" },
    { "graph_reindex": "reindex_graph" },
    { "deformable_conv_v1": "deformable_conv" },
    { "deformable_conv_v1_grad": "deformable_conv_grad" },
]
op_arg_name_mappings = [
    { 
        "abs":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "abs_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "accuracy":
        [
            { "Out":"x" },
            { "Indices":"indices" },
            { "Label":"label" },
            { "Accuracy":"accuracy" },
            { "Correct":"correct" },
            { "Total":"total" },
        ]
    },
    { 
        "acos":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "acosh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "acosh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "adadelta":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "AvgSquaredGrad":"avg_squared_grad" },
            { "AvgSquaredUpdate":"avg_squared_update" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "AvgSquaredGradOut":"moment_out" },
            { "AvgSquaredUpdateOut":"inf_norm_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "adagrad":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Moment":"moment" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "MomentOut":"moment_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "adam":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "Moment1":"moment1" },
            { "Moment2":"moment2" },
            { "Beta1Pow":"beta1_pow" },
            { "Beta2Pow":"beta2_pow" },
            { "MasterParam":"master_param" },
            { "SkipUpdate":"skip_update" },
            { "ParamOut":"param_out" },
            { "Moment1Out":"moment1_out" },
            { "Moment2Out":"moment2_out" },
            { "Beta1PowOut":"beta1_pow_out" },
            { "Beta2PowOut":"beta2_pow_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "adamax":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "Moment":"moment" },
            { "InfNorm":"inf_norm" },
            { "Beta1Pow":"beta1_pow" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "MomentOut":"moment_out" },
            { "InfNormOut":"inf_norm_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "adamw":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "Moment1":"moment1" },
            { "Moment2":"moment2" },
            { "Beta1Pow":"beta1_pow" },
            { "Beta2Pow":"beta2_pow" },
            { "MasterParam":"master_param" },
            { "SkipUpdate":"skip_update" },
            { "ParamOut":"param_out" },
            { "Moment1Out":"moment1_out" },
            { "Moment2Out":"moment2_out" },
            { "Beta1PowOut":"beta1_pow_out" },
            { "Beta2PowOut":"beta2_pow_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "elementwise_add":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Scale_x":"scale_x" },
            { "Scale_y":"scale_y" },
            { "Scale_out":"scale_out" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_add_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Scale_x":"scale_x" },
            { "Scale_y":"scale_y" },
            { "Scale_out":"scale_out" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_add_grad_grad":
        [
            { "Y":"y" },
            { "DOut":"grad_out" },
            { "DDX":"grad_x_grad" },
            { "DDY":"grad_y_grad" },
            { "DDOut":"grad_out_grad" },
        ]
    },
    { 
        "elementwise_add_triple_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Scale_x":"scale_x" },
            { "Scale_y":"scale_y" },
            { "Scale_out":"scale_out" },
            { "Out":"out" },
        ]
    },
    { 
        "sum":
        [
            { "X":"inputs" },
            { "Out":"out" },
        ]
    },
    { 
        "addmm":
        [
            { "Input":"input" },
            { "X":"x" },
            { "Y":"y" },
            { "Alpha":"alpha" },
            { "Beta":"beta" },
            { "Out":"out" },
        ]
    },
    { 
        "addmm_grad":
        [
            { "Input":"input" },
            { "X":"x" },
            { "Y":"y" },
            { "Alpha":"alpha" },
            { "Beta":"beta" },
            { "Out":"out" },
        ]
    },
    { 
        "affine_grid":
        [
            { "Theta":"input" },
            { "Output":"output" },
        ]
    },
    { 
        "affine_grid_grad":
        [
            { "Theta":"input" },
            { "Output":"output" },
        ]
    },
    { 
        "reduce_all":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "allclose":
        [
            { "Input":"x" },
            { "Other":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_amax":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_amax_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_amin":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_amin_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "angle":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "angle_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_any":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "range":
        [
            { "Start":"start" },
            { "End":"end" },
            { "Step":"step" },
            { "Out":"out" },
        ]
    },
    { 
        "arg_max":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "arg_min":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "argsort":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "tensor_array_to_tensor":
        [
            { "X":"x" },
            { "Out":"out" },
            { "OutIndex":"out_index" },
        ]
    },
    { 
        "tanh_shrink_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "OutIndex":"out_index" },
        ]
    },
    { 
        "as_complex":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "as_real":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "asin":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "asinh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "asinh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "assert":
        [
            { "Cond":"cond" },
            { "Data":"data" },
        ]
    },
    { 
        "assign":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "assign_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "assign_value":
        [
            { "Out":"out" },
        ]
    },
    { 
        "atan":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "atan2":
        [
            { "X1":"x" },
            { "X2":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "atan2_grad":
        [
            { "X1":"x" },
            { "X2":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "atanh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "atanh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "auc":
        [
            { "Predict":"x" },
            { "Label":"label" },
            { "StatPos":"stat_pos" },
            { "StatNeg":"stat_neg" },
            { "InsTagWeight":"ins_tag_weight" },
            { "AUC":"auc" },
            { "StatPosOut":"stat_pos_out" },
            { "StatNegOut":"stat_neg_out" },
        ]
    },
    { 
        "batch_norm":
        [
            { "X":"x" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "data_layout":"data_format" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "batch_norm_grad":
        [
            { "X":"x" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "data_layout":"data_format" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "batch_norm_grad_grad":
        [
            { "DScale":"scale_grad" },
            { "DX":"x_grad" },
            { "DDY":"grad_out_grad" },
            { "OutMean":"out_mean" },
            { "OutVariance":"out_variance" },
            { "DDX":"grad_x_grad" },
            { "DDScale":"grad_scale_grad" },
            { "DDBias":"grad_bias_grad" },
            { "DY":"grad_out" },
        ]
    },
    { 
        "bce_loss":
        [
            { "X":"input" },
            { "Label":"label" },
            { "Out":"out" },
        ]
    },
    { 
        "bce_loss_grad":
        [
            { "X":"input" },
            { "Label":"label" },
            { "Out":"out" },
        ]
    },
    { 
        "bernoulli":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "bicubic_interp_v2":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "bicubic_interp_v2_grad":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "bilinear_tensor_product":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Weight":"weight" },
            { "Bias":"bias" },
            { "Out":"out" },
        ]
    },
    { 
        "bilinear_tensor_product_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Weight":"weight" },
            { "Bias":"bias" },
            { "Out":"out" },
        ]
    },
    { 
        "bilinear_interp_v2":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "bilinear_interp_v2_grad":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "bincount":
        [
            { "X":"x" },
            { "Weights":"weights" },
            { "Out":"out" },
        ]
    },
    { 
        "bitwise_and":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "bitwise_not":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "bitwise_or":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "bitwise_xor":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "bmm":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "bn_act_xpu":
        [
            { "data_layout":"data_format" },
        ]
    },
    { 
        "box_coder":
        [
            { "PriorBox":"prior_box" },
            { "PriorBoxVar":"prior_box_var" },
            { "TargetBox":"target_box" },
            { "OutputBox":"output_box" },
        ]
    },
    { 
        "broadcast_tensors":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "broadcast_tensors_grad":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "c_concat":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_embedding":
        [
            { "W":"weight" },
            { "Ids":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_embedding_grad":
        [
            { "W":"weight" },
            { "Ids":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_softmax_with_cross_entropy":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "c_softmax_with_cross_entropy_grad":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "cast":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "ceil":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "ceil_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "celu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "celu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "celu_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "check_finite_and_unscale":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Out":"out" },
            { "FoundInfinite":"found_infinite" },
        ]
    },
    { 
        "cholesky":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cholesky_solve":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "class_center_sample":
        [
            { "Label":"label" },
            { "RemappedLabel":"remapped_label" },
            { "SampledLocalClassCenter":"sampled_local_class_center" },
        ]
    },
    { 
        "clip":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "clip_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "clip_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "clip_by_norm":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "coalesce_tensor":
        [
            { "Input":"input" },
            { "user_defined_size_of_dtype":"size_of_dtype" },
            { "Output":"output" },
            { "FusedOutput":"fused_output" },
        ]
    },
    { 
        "complex":
        [
            { "X":"real" },
            { "Y":"imag" },
            { "Out":"out" },
        ]
    },
    { 
        "complex_grad":
        [
            { "X":"real" },
            { "Y":"imag" },
            { "Out":"out" },
        ]
    },
    { 
        "concat":
        [
            { "X":"x" },
            { "axis":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "concat_grad":
        [
            { "X":"x" },
            { "axis":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "concat_double_grad":
        [
            { "X":"x" },
            { "axis":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "conj":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "conv2d":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv2d_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv2d_grad_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv2d_transpose":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "Output":"out" },
        ]
    },
    { 
        "conv2d_transpose_grad":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "Output":"out" },
        ]
    },
    { 
        "conv2d_transpose_grad_grad":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "Output":"out" },
        ]
    },
    { 
        "conv3d":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv3d_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv3d_grad_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv3d_transpose":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "conv3d_transpose_grad":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Output":"out" },
        ]
    },
    { 
        "cos":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cos_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cos_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cos_triple_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cosh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cosh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "crop_tensor":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "crop_tensor_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cross":
        [
            { "X":"x" },
            { "Y":"y" },
            { "dim":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "softmax_with_cross_entropy":
        [
            { "Logits":"input" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "softmax_with_cross_entropy_grad":
        [
            { "Logits":"input" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "cumprod":
        [
            { "X":"x" },
            { "dim":"dim" },
            { "Out":"out" },
        ]
    },
    { 
        "cumprod_grad":
        [
            { "X":"x" },
            { "dim":"dim" },
            { "Out":"out" },
        ]
    },
    { 
        "cumsum":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "cumsum_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "decode_jpeg":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "deformable_conv":
        [
            { "Input":"x" },
            { "Offset":"offset" },
            { "Filter":"filter" },
            { "Mask":"mask" },
            { "Output":"out" },
        ]
    },
    { 
        "deformable_conv_grad":
        [
            { "Input":"x" },
            { "Offset":"offset" },
            { "Filter":"filter" },
            { "Mask":"mask" },
            { "Output":"out" },
        ]
    },
    { 
        "depthwise_conv2d":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_in_eltwise":"scale_in_eltwise" },
            { "Scale_weights":"scale_weights" },
            { "Output":"out" },
        ]
    },
    { 
        "depthwise_conv2d_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_in_eltwise":"scale_in_eltwise" },
            { "Scale_weights":"scale_weights" },
            { "Output":"out" },
        ]
    },
    { 
        "depthwise_conv2d_grad_grad":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_in_eltwise":"scale_in_eltwise" },
            { "Scale_weights":"scale_weights" },
            { "Output":"out" },
        ]
    },
    { 
        "depthwise_conv2d_transpose":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "Output":"out" },
        ]
    },
    { 
        "depthwise_conv2d_transpose_grad":
        [
            { "Input":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "Output":"out" },
        ]
    },
    { 
        "dequantize":
        [
            { "Input":"input" },
            { "Scale":"scale" },
            { "Shift":"shift" },
            { "Output":"output" },
        ]
    },
    { 
        "determinant":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "determinant_grad":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "diag_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "diag_v2_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "diag_embed":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "diagonal":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "digamma":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "dirichlet":
        [
            { "Alpha":"alpha" },
            { "Out":"out" },
        ]
    },
    { 
        "dist":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_div":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_div_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "dot":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "dropout":
        [
            { "X":"x" },
            { "Seed":"seed_tensor" },
            { "dropout_prob":"p" },
            { "is_test":"is_test" },
            { "dropout_implementation":"mode" },
            { "seed":"seed" },
            { "fix_seed":"fix_seed" },
            { "Out":"out" },
            { "Mask":"mask" },
        ]
    },
    { 
        "dropout_grad":
        [
            { "X":"x" },
            { "Seed":"seed_tensor" },
            { "dropout_prob":"p" },
            { "is_test":"is_test" },
            { "dropout_implementation":"mode" },
            { "seed":"seed" },
            { "fix_seed":"fix_seed" },
            { "Out":"out" },
            { "Mask":"mask" },
        ]
    },
    { 
        "edit_distance":
        [
            { "Hyps":"hyps" },
            { "Refs":"refs" },
            { "HypsLength":"hypslength" },
            { "RefsLength":"refslength" },
            { "SequenceNum":"sequencenum" },
            { "Out":"out" },
        ]
    },
    { 
        "eig":
        [
            { "X":"x" },
            { "Eigenvalues":"out_w" },
            { "Eigenvectors":"out_v" },
        ]
    },
    { 
        "eigh":
        [
            { "X":"x" },
            { "Eigenvalues":"out_w" },
            { "Eigenvectors":"out_v" },
        ]
    },
    { 
        "eigvals":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "eigvalsh":
        [
            { "X":"x" },
            { "UPLO":"uplo" },
            { "Eigenvalues":"eigenvalues" },
            { "Eigenvectors":"eigenvectors" },
        ]
    },
    { 
        "eigvalsh_grad":
        [
            { "X":"x" },
            { "UPLO":"uplo" },
            { "Eigenvalues":"eigenvalues" },
            { "Eigenvectors":"eigenvectors" },
        ]
    },
    { 
        "einsum":
        [
            { "Operands":"x" },
            { "Out":"out" },
            { "InnerCache":"inner_cache" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "einsum_grad":
        [
            { "Operands":"x" },
            { "Out":"out" },
            { "InnerCache":"inner_cache" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "elementwise_pow":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_pow_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elu_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "lookup_table_v2":
        [
            { "Ids":"x" },
            { "W":"weight" },
            { "is_sparse":"sparse" },
            { "Out":"out" },
        ]
    },
    { 
        "lookup_table_v2_grad":
        [
            { "Ids":"x" },
            { "W":"weight" },
            { "is_sparse":"sparse" },
            { "Out":"out" },
        ]
    },
    { 
        "empty":
        [
            { "Out":"out" },
        ]
    },
    { 
        "equal":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "equal_all":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "erf":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "erfinv":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "exp":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "exp_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "expand_v2":
        [
            { "X":"x" },
            { "shape":"shape" },
            { "Out":"out" },
        ]
    },
    { 
        "expand_v2_grad":
        [
            { "X":"x" },
            { "shape":"shape" },
            { "Out":"out" },
        ]
    },
    { 
        "expand_v2_double_grad":
        [
            { "X":"x" },
            { "shape":"shape" },
            { "Out":"out" },
        ]
    },
    { 
        "expand_as_v2":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "expand_as_v2_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "expm1":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "expm1_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "exponential":
        [
            { "X":"x" },
            { "lambda":"lam" },
            { "Out":"out" },
        ]
    },
    { 
        "exponential_grad":
        [
            { "X":"x" },
            { "lambda":"lam" },
            { "Out":"out" },
        ]
    },
    { 
        "eye":
        [
            { "Out":"out" },
        ]
    },
    { 
        "fc":
        [
            { "Input":"input" },
            { "W":"w" },
            { "Bias":"bias" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_weights":"scale_weights" },
            { "Out":"out" },
        ]
    },
    { 
        "feed":
        [
            { "Out":"out" },
        ]
    },
    { 
        "fft_c2c":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fft_c2r":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fft_r2c":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_any":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_any_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_diagonal":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_diagonal_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_diagonal_tensor":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "flatten_contiguous_range":
        [
            { "X":"x" },
            { "start_axis":"start_axis" },
            { "stop_axis":"stop_axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "flatten_contiguous_range_grad":
        [
            { "X":"x" },
            { "start_axis":"start_axis" },
            { "stop_axis":"stop_axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "flip":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "floor":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "floor_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_floordiv":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_fmax":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_fmax_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_fmin":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_fmin_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "fold":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "frame":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "frame_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "frobenius_norm":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "frobenius_norm_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "fill_constant":
        [
            { "Out":"out" },
        ]
    },
    { 
        "fill_any_like":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "fused_attention":
        [
            { "X":"x" },
            { "LnScale":"ln_scale" },
            { "LnBias":"ln_bias" },
            { "QKVW":"qkv_weight" },
            { "QKVBias":"qkv_bias" },
            { "CacheKV":"cache_kv" },
            { "SrcMask":"src_mask" },
            { "OutLinearW":"out_linear_weight" },
            { "OutLinearBias":"out_linear_bias" },
            { "Ln2Scale":"ln_scale_2" },
            { "Ln2Bias":"ln_bias_2" },
            { "LnMean":"ln_mean" },
            { "LnVariance":"ln_var" },
            { "LnOut":"ln_out" },
            { "QKVOut":"qkv_out" },
            { "QKVBiasOut":"qkv_bias_out" },
            { "TransposeOut2":"transpose_out_2" },
            { "QKOut":"qk_out" },
            { "QKTVOut":"qktv_out" },
            { "SoftmaxOut":"softmax_out" },
            { "AttnDropoutMaskOut":"attn_dropout_mask_out" },
            { "AttnDropoutOut":"attn_dropout_out" },
            { "SrcMaskOut":"src_mask_out" },
            { "FMHAOut":"fmha_out" },
            { "OutLinearOut":"out_linear_out" },
            { "DropoutMaskOut":"dropout_mask_out" },
            { "Ln2Mean":"ln_mean_2" },
            { "Ln2Variance":"ln_var_2" },
            { "BiasDropoutResidualOut":"bias_dropout_residual_out" },
            { "CacheKVOut":"cache_kv_out" },
            { "Y":"out" },
        ]
    },
    { 
        "fused_attention_grad":
        [
            { "X":"x" },
            { "LnScale":"ln_scale" },
            { "LnBias":"ln_bias" },
            { "QKVW":"qkv_weight" },
            { "QKVBias":"qkv_bias" },
            { "CacheKV":"cache_kv" },
            { "SrcMask":"src_mask" },
            { "OutLinearW":"out_linear_weight" },
            { "OutLinearBias":"out_linear_bias" },
            { "Ln2Scale":"ln_scale_2" },
            { "Ln2Bias":"ln_bias_2" },
            { "LnMean":"ln_mean" },
            { "LnVariance":"ln_var" },
            { "LnOut":"ln_out" },
            { "QKVOut":"qkv_out" },
            { "QKVBiasOut":"qkv_bias_out" },
            { "TransposeOut2":"transpose_out_2" },
            { "QKOut":"qk_out" },
            { "QKTVOut":"qktv_out" },
            { "SoftmaxOut":"softmax_out" },
            { "AttnDropoutMaskOut":"attn_dropout_mask_out" },
            { "AttnDropoutOut":"attn_dropout_out" },
            { "SrcMaskOut":"src_mask_out" },
            { "FMHAOut":"fmha_out" },
            { "OutLinearOut":"out_linear_out" },
            { "DropoutMaskOut":"dropout_mask_out" },
            { "Ln2Mean":"ln_mean_2" },
            { "Ln2Variance":"ln_var_2" },
            { "BiasDropoutResidualOut":"bias_dropout_residual_out" },
            { "CacheKVOut":"cache_kv_out" },
            { "Y":"out" },
        ]
    },
    { 
        "fused_batch_norm_act":
        [
            { "X":"x" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "fused_batch_norm_act_grad":
        [
            { "X":"x" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "fused_bias_dropout_residual_layer_norm":
        [
            { "X":"x" },
            { "Residual":"residual" },
            { "Bias":"bias" },
            { "LnScale":"ln_scale" },
            { "LnBias":"ln_bias" },
            { "BiasDropoutResidualOut":"bias_dropout_residual_out" },
            { "DropoutMaskOut":"dropout_mask_out" },
            { "LnMean":"ln_mean" },
            { "LnVariance":"ln_variance" },
            { "Y":"y" },
        ]
    },
    { 
        "fused_bias_dropout_residual_layer_norm_grad":
        [
            { "X":"x" },
            { "Residual":"residual" },
            { "Bias":"bias" },
            { "LnScale":"ln_scale" },
            { "LnBias":"ln_bias" },
            { "BiasDropoutResidualOut":"bias_dropout_residual_out" },
            { "DropoutMaskOut":"dropout_mask_out" },
            { "LnMean":"ln_mean" },
            { "LnVariance":"ln_variance" },
            { "Y":"y" },
        ]
    },
    { 
        "fused_bn_add_activation":
        [
            { "X":"x" },
            { "Z":"z" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "fused_bn_add_activation_grad":
        [
            { "X":"x" },
            { "Z":"z" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "fused_conv2d":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "ResidualData":"residual_param" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_in_eltwise":"scale_in_eltwise" },
            { "Scale_weights":"scale_weights" },
            { "Output":"output" },
        ]
    },
    { 
        "fused_conv2d_add_act":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "ResidualData":"residual_data" },
            { "Output":"output" },
            { "Outputs":"outputs" },
        ]
    },
    { 
        "fused_conv3d":
        [
            { "Input":"input" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "ResidualData":"residual_param" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Scale_in_eltwise":"scale_in_eltwise" },
            { "Scale_weights":"scale_weights" },
            { "Output":"output" },
        ]
    },
    { 
        "fused_embedding_eltwise_layernorm":
        [
            { "Ids":"ids" },
            { "Embs":"embs" },
            { "Bias":"bias" },
            { "Scale":"scale" },
            { "Out":"out" },
        ]
    },
    { 
        "fused_fc_elementwise_layernorm":
        [
            { "X":"x" },
            { "W":"w" },
            { "Y":"y" },
            { "Bias0":"bias0" },
            { "Scale":"scale" },
            { "Bias1":"bias1" },
            { "Out":"out" },
            { "Mean":"mean" },
            { "Variance":"variance" },
        ]
    },
    { 
        "fused_feedforward":
        [
            { "X":"x" },
            { "Dropout1Seed":"dropout1_seed" },
            { "Dropout2Seed":"dropout2_seed" },
            { "Linear1Weight":"linear1_weight" },
            { "Linear1Bias":"linear1_bias" },
            { "Linear2Weight":"linear2_weight" },
            { "Linear2Bias":"linear2_bias" },
            { "Ln1Scale":"ln1_scale" },
            { "Ln1Bias":"ln1_bias" },
            { "Ln2Scale":"ln2_scale" },
            { "Ln2Bias":"ln2_bias" },
            { "dropout1_seed":"dropout1_seed_val" },
            { "dropout2_seed":"dropout2_seed_val" },
            { "dropout1_rate":"dropout1_prob" },
            { "dropout2_rate":"dropout2_prob" },
            { "Out":"out" },
            { "Dropout1Mask":"dropout1_mask" },
            { "Dropout2Mask":"dropout2_mask" },
            { "Ln1Mean":"ln1_mean" },
            { "Ln1Variance":"ln1_variance" },
            { "Ln2Mean":"ln2_mean" },
            { "Ln2Variance":"ln2_variance" },
            { "Linear1Out":"linear1_out" },
            { "Ln1Out":"ln1_out" },
            { "Dropout1Out":"dropout1_out" },
            { "Dropout2Out":"dropout2_out" },
        ]
    },
    { 
        "fused_feedforward_grad":
        [
            { "X":"x" },
            { "Dropout1Seed":"dropout1_seed" },
            { "Dropout2Seed":"dropout2_seed" },
            { "Linear1Weight":"linear1_weight" },
            { "Linear1Bias":"linear1_bias" },
            { "Linear2Weight":"linear2_weight" },
            { "Linear2Bias":"linear2_bias" },
            { "Ln1Scale":"ln1_scale" },
            { "Ln1Bias":"ln1_bias" },
            { "Ln2Scale":"ln2_scale" },
            { "Ln2Bias":"ln2_bias" },
            { "dropout1_seed":"dropout1_seed_val" },
            { "dropout2_seed":"dropout2_seed_val" },
            { "dropout1_rate":"dropout1_prob" },
            { "dropout2_rate":"dropout2_prob" },
            { "Out":"out" },
            { "Dropout1Mask":"dropout1_mask" },
            { "Dropout2Mask":"dropout2_mask" },
            { "Ln1Mean":"ln1_mean" },
            { "Ln1Variance":"ln1_variance" },
            { "Ln2Mean":"ln2_mean" },
            { "Ln2Variance":"ln2_variance" },
            { "Linear1Out":"linear1_out" },
            { "Ln1Out":"ln1_out" },
            { "Dropout1Out":"dropout1_out" },
            { "Dropout2Out":"dropout2_out" },
        ]
    },
    { 
        "fused_gemm_epilogue":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Bias":"bias" },
            { "Out":"out" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "fused_gemm_epilogue_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "ReserveSpace":"reserve_space" },
            { "DOut":"out_grad" },
            { "DX":"x_grad" },
            { "DY":"y_grad" },
            { "DBias":"bias_grad" },
        ]
    },
    { 
        "fusion_gru":
        [
            { "X":"x" },
            { "H0":"h0" },
            { "WeightX":"weight_x" },
            { "WeightH":"weight_h" },
            { "Bias":"bias" },
            { "Scale_data":"scale_data" },
            { "Shift_data":"shift_data" },
            { "Scale_weights":"scale_weights" },
            { "ReorderedH0":"reordered_h0" },
            { "XX":"xx" },
            { "BatchedInput":"batched_input" },
            { "BatchedOut":"batched_out" },
            { "Hidden":"hidden" },
        ]
    },
    { 
        "fusion_repeated_fc_relu":
        [
            { "X":"x" },
            { "W":"w" },
            { "Bias":"bias" },
            { "ReluOut":"relu_out" },
            { "Out":"out" },
        ]
    },
    { 
        "fusion_seqconv_eltadd_relu":
        [
            { "X":"x" },
            { "Filter":"filter" },
            { "Bias":"bias" },
            { "contextLength":"context_length" },
            { "contextStart":"context_start" },
            { "contextStride":"context_stride" },
            { "Out":"out" },
            { "ColMat":"col_mat" },
        ]
    },
    { 
        "fusion_seqexpand_concat_fc":
        [
            { "X":"x" },
            { "FCWeight":"fc_weight" },
            { "FCBias":"fc_bias" },
            { "Out":"out" },
            { "FCOut":"fc_out" },
        ]
    },
    { 
        "fusion_transpose_flatten_concat":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "gather":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "gather_grad":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "gather_nd":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "gather_nd_grad":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "gather_tree":
        [
            { "Ids":"ids" },
            { "Parents":"parents" },
            { "Out":"out" },
        ]
    },
    { 
        "gaussian_random":
        [
            { "Out":"out" },
        ]
    },
    { 
        "gelu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "gelu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "generate_proposals_v2":
        [
            { "Scores":"scores" },
            { "BboxDeltas":"bbox_deltas" },
            { "ImShape":"im_shape" },
            { "Anchors":"anchors" },
            { "Variances":"variances" },
            { "pre_nms_topN":"pre_nms_top_n" },
            { "post_nms_topN":"post_nms_top_n" },
            { "RpnRois":"rpn_rois" },
            { "RpnRoiProbs":"rpn_roi_probs" },
            { "RpnRoisNum":"rpn_rois_num" },
        ]
    },
    { 
        "grad_add":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "graph_khop_sampler":
        [
            { "Row":"row" },
            { "Col_Ptr":"colptr" },
            { "X":"x" },
            { "Eids":"eids" },
            { "Out_Src":"out_src" },
            { "Out_Dst":"out_dst" },
            { "Sample_Index":"sample_index" },
            { "Reindex_X":"reindex_x" },
            { "Out_Eids":"out_eids" },
        ]
    },
    { 
        "graph_sample_neighbors":
        [
            { "Row":"row" },
            { "Col_Ptr":"colptr" },
            { "X":"x" },
            { "Eids":"eids" },
            { "Perm_Buffer":"perm_buffer" },
            { "Out":"out" },
            { "Out_Count":"out_count" },
            { "Out_Eids":"out_eids" },
        ]
    },
    { 
        "greater_equal":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "greater_than":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "grid_sampler":
        [
            { "X":"x" },
            { "Grid":"grid" },
            { "Output":"out" },
        ]
    },
    { 
        "grid_sampler_grad":
        [
            { "X":"x" },
            { "Grid":"grid" },
            { "Output":"out" },
        ]
    },
    { 
        "group_norm":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "data_layout":"data_format" },
            { "Y":"y" },
            { "Mean":"mean" },
            { "Variance":"variance" },
        ]
    },
    { 
        "gumbel_softmax":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_shrink":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_shrink_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_sigmoid":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_sigmoid_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_swish":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "hard_swish_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "brelu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "brelu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_heaviside":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_heaviside_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "histogram":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "hierarchical_sigmoid":
        [
            { "X":"x" },
            { "W":"w" },
            { "Label":"label" },
            { "Bias":"bias" },
            { "PathTable":"path" },
            { "PathCode":"code" },
            { "Out":"out" },
            { "PreOut":"pre_out" },
            { "W_Out":"w_out" },
        ]
    },
    { 
        "hierarchical_sigmoid_grad":
        [
            { "X":"x" },
            { "W":"w" },
            { "Label":"label" },
            { "Bias":"bias" },
            { "PathTable":"path" },
            { "PathCode":"code" },
            { "Out":"out" },
            { "PreOut":"pre_out" },
            { "W_Out":"w_out" },
        ]
    },
    { 
        "huber_loss":
        [
            { "X":"input" },
            { "Y":"label" },
            { "Out":"out" },
            { "Residual":"residual" },
        ]
    },
    { 
        "huber_loss_grad":
        [
            { "X":"input" },
            { "Y":"label" },
            { "Out":"out" },
            { "Residual":"residual" },
        ]
    },
    { 
        "imag":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "imag_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "increment":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "index_add":
        [
            { "X":"x" },
            { "Index":"index" },
            { "AddValue":"add_value" },
            { "Out":"out" },
        ]
    },
    { 
        "index_sample":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "index_select":
        [
            { "X":"x" },
            { "Index":"index" },
            { "dim":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "instance_norm":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"y" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
        ]
    },
    { 
        "inverse":
        [
            { "Input":"x" },
            { "Output":"out" },
        ]
    },
    { 
        "is_empty":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "isclose":
        [
            { "Input":"x" },
            { "Other":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "isfinite_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "isinf_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "isnan_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "kldiv_loss":
        [
            { "X":"x" },
            { "Target":"label" },
            { "Loss":"out" },
        ]
    },
    { 
        "kldiv_loss_grad":
        [
            { "X":"x" },
            { "Target":"label" },
            { "Loss":"out" },
        ]
    },
    { 
        "kron":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "kron_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "kthvalue":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "label_smooth":
        [
            { "X":"label" },
            { "PriorDist":"prior_dist" },
            { "Out":"out" },
        ]
    },
    { 
        "lamb":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "Moment1":"moment1" },
            { "Moment2":"moment2" },
            { "Beta1Pow":"beta1_pow" },
            { "Beta2Pow":"beta2_pow" },
            { "MasterParam":"master_param" },
            { "SkipUpdate":"skip_update" },
            { "ParamOut":"param_out" },
            { "Moment1Out":"moment1_out" },
            { "Moment2Out":"moment2_out" },
            { "Beta1PowOut":"beta1_pow_out" },
            { "Beta2PowOut":"beta2_pow_out" },
            { "MasterParamOut":"master_param_outs" },
        ]
    },
    { 
        "layer_norm":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "Mean":"mean" },
            { "Variance":"variance" },
        ]
    },
    { 
        "layer_norm_grad":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Y":"out" },
            { "Mean":"mean" },
            { "Variance":"variance" },
        ]
    },
    { 
        "leaky_relu":
        [
            { "X":"x" },
            { "alpha":"negative_slope" },
            { "Out":"out" },
        ]
    },
    { 
        "leaky_relu_grad":
        [
            { "X":"x" },
            { "alpha":"negative_slope" },
            { "Out":"out" },
        ]
    },
    { 
        "leaky_relu_grad_grad":
        [
            { "X":"x" },
            { "alpha":"negative_slope" },
            { "Out":"out" },
        ]
    },
    { 
        "lerp":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Weight":"weight" },
            { "Out":"out" },
        ]
    },
    { 
        "lerp_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Weight":"weight" },
            { "Out":"out" },
        ]
    },
    { 
        "less_equal":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "less_than":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "lgamma":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "linear_interp_v2":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "linear_interp_v2_grad":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "linspace":
        [
            { "Start":"start" },
            { "Stop":"stop" },
            { "Num":"number" },
            { "Out":"out" },
        ]
    },
    { 
        "log":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log10":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log10_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log1p":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log1p_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log2_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log_loss":
        [
            { "Predicted":"input" },
            { "Labels":"label" },
            { "Loss":"out" },
        ]
    },
    { 
        "log_loss_grad":
        [
            { "Predicted":"input" },
            { "Labels":"label" },
            { "Loss":"out" },
        ]
    },
    { 
        "log_softmax":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "log_softmax_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logcumsumexp":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logcumsumexp_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logical_and":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "logical_not":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logical_or":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "logical_xor":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "logit":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logsigmoid":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logsigmoid_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logsumexp":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logsumexp_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "lrn":
        [
            { "X":"x" },
            { "Out":"out" },
            { "MidOut":"mid_out" },
        ]
    },
    { 
        "lrn_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "MidOut":"mid_out" },
        ]
    },
    { 
        "lstsq":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Solution":"solution" },
            { "Residuals":"residuals" },
            { "Rank":"rank" },
            { "SingularValues":"singular_values" },
        ]
    },
    { 
        "lu_unpack":
        [
            { "X":"x" },
            { "Pivots":"y" },
            { "Pmat":"pmat" },
            { "L":"l" },
            { "U":"u" },
        ]
    },
    { 
        "lu_unpack_grad":
        [
            { "X":"x" },
            { "Pivots":"y" },
            { "Pmat":"pmat" },
            { "L":"l" },
            { "U":"u" },
        ]
    },
    { 
        "margin_cross_entropy":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "margin_cross_entropy_grad":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "Softmax":"softmax" },
            { "Loss":"loss" },
        ]
    },
    { 
        "masked_select":
        [
            { "X":"x" },
            { "Mask":"mask" },
            { "Y":"out" },
        ]
    },
    { 
        "matmul_v2":
        [
            { "X":"x" },
            { "Y":"y" },
            { "trans_x":"transpose_x" },
            { "trans_y":"transpose_y" },
            { "Out":"out" },
        ]
    },
    { 
        "matmul_v2_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "trans_x":"transpose_x" },
            { "trans_y":"transpose_y" },
            { "Out":"out" },
        ]
    },
    { 
        "matmul_v2_grad_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "trans_x":"transpose_x" },
            { "trans_y":"transpose_y" },
            { "Out":"out" },
        ]
    },
    { 
        "matmul_v2_triple_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "trans_x":"transpose_x" },
            { "trans_y":"transpose_y" },
            { "Out":"out" },
        ]
    },
    { 
        "mul":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "mul_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "matrix_nms":
        [
            { "BBoxes":"bboxes" },
            { "Scores":"scores" },
            { "Out":"out" },
            { "Index":"index" },
            { "RoisNum":"roisnum" },
        ]
    },
    { 
        "matrix_power":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "matrix_rank":
        [
            { "X":"x" },
            { "TolTensor":"atol_tensor" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_max":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_max_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "max_pool2d_with_index":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
            { "Mask":"mask" },
        ]
    },
    { 
        "max_pool3d_with_index":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
            { "Mask":"mask" },
        ]
    },
    { 
        "elementwise_max":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_max_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "maxout":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_mean":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_mean_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "mean":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "mean_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "merge_selected_rows":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "merged_adam_":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "Moment1":"moment1" },
            { "Moment2":"moment2" },
            { "Beta1Pow":"beta1_pow" },
            { "Beta2Pow":"beta2_pow" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "Moment1Out":"moment1_out" },
            { "Moment2Out":"moment2_out" },
            { "Beta1PowOut":"beta1_pow_out" },
            { "Beta2PowOut":"beta2_pow_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "merged_momentum":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Velocity":"velocity" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "VelocityOut":"velocity_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "meshgrid":
        [
            { "X":"inputs" },
            { "Out":"out" },
        ]
    },
    { 
        "meshgrid_grad":
        [
            { "X":"inputs" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_min":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_min_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_min":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_min_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "mish":
        [
            { "X":"x" },
            { "threshold":"lambda" },
            { "Out":"out" },
        ]
    },
    { 
        "mish_grad":
        [
            { "X":"x" },
            { "threshold":"lambda" },
            { "Out":"out" },
        ]
    },
    { 
        "mode":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "mode_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "momentum":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Velocity":"velocity" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "VelocityOut":"velocity_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "multi_dot":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "multi_dot_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "multi_gru":
        [
            { "X":"x" },
            { "WeightX":"weight_x" },
            { "WeightH":"weight_h" },
            { "Bias":"bias" },
            { "Scale_weights":"scale_weights" },
            { "Scale_data":"scale_data" },
            { "Shift_data":"shift_data" },
            { "Hidden":"hidden" },
        ]
    },
    { 
        "multiclass_nms3":
        [
            { "BBoxes":"bboxes" },
            { "Scores":"scores" },
            { "RoisNum":"rois_num" },
            { "Out":"out" },
            { "Index":"index" },
            { "NmsRoisNum":"nms_rois_num" },
        ]
    },
    { 
        "multihead_matmul":
        [
            { "Input":"input" },
            { "W":"w" },
            { "Bias":"bias" },
            { "BiasQK":"bias_qk" },
            { "transpose_Q":"transpose_q" },
            { "transpose_K":"transpose_k" },
            { "transpose_V":"transpose_v" },
            { "Out":"out" },
        ]
    },
    { 
        "multinomial":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "multiplex":
        [
            { "X":"inputs" },
            { "Ids":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "multiplex_grad":
        [
            { "X":"inputs" },
            { "Ids":"index" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_mul":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_mul_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "mv":
        [
            { "X":"x" },
            { "Vec":"vec" },
            { "Out":"out" },
        ]
    },
    { 
        "nanmedian":
        [
            { "X":"x" },
            { "Out":"out" },
            { "MedianIndex":"medians" },
        ]
    },
    { 
        "nanmedian_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "MedianIndex":"medians" },
        ]
    },
    { 
        "nearest_interp_v2":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "nearest_interp_v2_grad":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "nll_loss":
        [
            { "X":"input" },
            { "Label":"label" },
            { "Weight":"weight" },
            { "Out":"out" },
            { "Total_weight":"total_weight" },
        ]
    },
    { 
        "nll_loss_grad":
        [
            { "X":"input" },
            { "Label":"label" },
            { "Weight":"weight" },
            { "Out":"out" },
            { "Total_weight":"total_weight" },
        ]
    },
    { 
        "nms":
        [
            { "Boxes":"x" },
            { "iou_threshold":"threshold" },
            { "KeepBoxesIdxs":"out" },
        ]
    },
    { 
        "where_index":
        [
            { "Condition":"condition" },
            { "Out":"out" },
        ]
    },
    { 
        "norm":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Norm":"norm" },
        ]
    },
    { 
        "norm_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Norm":"norm" },
        ]
    },
    { 
        "not_equal":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "size":
        [
            { "Input":"x" },
            { "Out":"size" },
        ]
    },
    { 
        "one_hot_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "overlap_add":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "overlap_add_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "p_norm":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "p_norm_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pad_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pad3d":
        [
            { "X":"x" },
            { "value":"pad_value" },
            { "Out":"out" },
        ]
    },
    { 
        "pad3d_grad":
        [
            { "X":"x" },
            { "value":"pad_value" },
            { "Out":"out" },
        ]
    },
    { 
        "pad3d_double_grad":
        [
            { "X":"x" },
            { "value":"pad_value" },
            { "Out":"out" },
        ]
    },
    { 
        "pixel_shuffle":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pixel_shuffle_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pixel_unshuffle":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pixel_unshuffle_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "poisson":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "pool2d":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
        ]
    },
    { 
        "pool2d_grad":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
        ]
    },
    { 
        "pool2d_double_grad":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
        ]
    },
    { 
        "pool3d":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
        ]
    },
    { 
        "pool3d_grad":
        [
            { "X":"x" },
            { "ksize":"kernel_size" },
            { "Out":"out" },
        ]
    },
    { 
        "pow":
        [
            { "X":"x" },
            { "factor":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "pow_grad":
        [
            { "X":"x" },
            { "factor":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "pow_double_grad":
        [
            { "X":"x" },
            { "factor":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "pow_triple_grad":
        [
            { "X":"x" },
            { "factor":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "prelu":
        [
            { "X":"x" },
            { "Alpha":"alpha" },
            { "Out":"out" },
        ]
    },
    { 
        "prelu_grad":
        [
            { "X":"x" },
            { "Alpha":"alpha" },
            { "Out":"out" },
        ]
    },
    { 
        "print":
        [
            { "In":"in" },
            { "Out":"out" },
        ]
    },
    { 
        "prior_box":
        [
            { "Input":"input" },
            { "Image":"image" },
            { "Boxes":"out" },
            { "Variances":"var" },
        ]
    },
    { 
        "reduce_prod":
        [
            { "X":"x" },
            { "dim":"dims" },
            { "keep_dim":"keep_dim" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_prod_grad":
        [
            { "X":"x" },
            { "dim":"dims" },
            { "keep_dim":"keep_dim" },
            { "Out":"out" },
        ]
    },
    { 
        "psroi_pool":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
        ]
    },
    { 
        "psroi_pool_grad":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
        ]
    },
    { 
        "push_sparse_v2":
        [
            { "Ids":"x" },
            { "w":"W" },
            { "Out":"out" },
        ]
    },
    { 
        "put_along_axis":
        [
            { "Input":"arr" },
            { "Index":"indices" },
            { "Value":"values" },
            { "Axis":"axis" },
            { "Reduce":"reduce" },
            { "Include_self":"include_self" },
            { "Result":"out" },
        ]
    },
    { 
        "put_along_axis_grad":
        [
            { "Input":"arr" },
            { "Index":"indices" },
            { "Value":"values" },
            { "Axis":"axis" },
            { "Reduce":"reduce" },
            { "Include_self":"include_self" },
            { "Result":"out" },
        ]
    },
    { 
        "qr":
        [
            { "X":"x" },
            { "Q":"q" },
            { "R":"r" },
        ]
    },
    { 
        "qr_grad":
        [
            { "X":"x" },
            { "Q":"q" },
            { "R":"r" },
        ]
    },
    { 
        "quantize":
        [
            { "Input":"input" },
            { "Scale":"scale" },
            { "Shift":"shift" },
            { "Include_self":"include_self" },
            { "Output":"output" },
        ]
    },
    { 
        "randint":
        [
            { "Out":"out" },
        ]
    },
    { 
        "randperm":
        [
            { "Out":"out" },
        ]
    },
    { 
        "real":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "real_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "reciprocal":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "reciprocal_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "relu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "relu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "relu_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "relu6":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "relu6_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_mod":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "renorm":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "renorm_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "repeat_interleave":
        [
            { "X":"x" },
            { "Repeats":"repeats" },
            { "Out":"out" },
            { "dim":"axis" },
        ]
    },
    { 
        "repeat_interleave_grad":
        [
            { "X":"x" },
            { "Repeats":"repeats" },
            { "dim":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "repeat_interleave_with_tensor_index":
        [
            { "X":"x" },
            { "RepeatTensor":"repeats" },
            { "dim":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "repeat_interleave_with_tensor_index_grad":
        [
            { "X":"x" },
            { "RepeatTensor":"repeats" },
            { "dim":"axis" },
            { "Out":"out" },
        ]
    },
    { 
        "requantize":
        [
            { "Input":"input" },
            { "Scale_in":"scale_in" },
            { "Scale_out":"scale_out" },
            { "Shift_in":"shift_in" },
            { "Shift_out":"shift_out" },
            { "Output":"output" },
        ]
    },
    { 
        "reshape2":
        [
            { "X":"x" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "reshape2_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "reverse":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "rmsprop":
        [
            { "Param":"param" },
            { "MeanSquare":"mean_square" },
            { "MeanGrad":"mean_grad" },
            { "LearningRate":"learning_rate" },
            { "Grad":"grad" },
            { "Moment":"moment" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "MomentOut":"moment_out" },
            { "MeanSquareOut":"mean_square_out" },
            { "MeanGradOut":"mean_grad_out" },
            { "MasterParamOut":"master_param_outs" },
        ]
    },
    { 
        "rnn":
        [
            { "Input":"x" },
            { "PreState":"pre_state" },
            { "WeightList":"weight_list" },
            { "SequenceLength":"sequence_length" },
            { "Out":"out" },
            { "DropoutState":"dropout_state_out" },
            { "State":"state" },
            { "Reserve":"reserve" },
        ]
    },
    { 
        "rnn_grad":
        [
            { "Input":"x" },
            { "PreState":"pre_state" },
            { "WeightList":"weight_list" },
            { "SequenceLength":"sequence_length" },
            { "Out":"out" },
            { "DropoutState":"dropout_state_out" },
            { "State":"state" },
            { "Reserve":"reserve" },
        ]
    },
    { 
        "roi_align":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
        ]
    },
    { 
        "roi_align_grad":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
        ]
    },
    { 
        "roi_pool":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
            { "Argmax":"arg_max" },
        ]
    },
    { 
        "roi_pool_grad":
        [
            { "X":"x" },
            { "ROIs":"boxes" },
            { "RoisNum":"boxes_num" },
            { "Out":"out" },
            { "Argmax":"arg_max" },
        ]
    },
    { 
        "roll":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "roll_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "round":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "round_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "row_conv":
        [
            { "X":"x" },
            { "Filter":"filter" },
            { "Out":"out" },
        ]
    },
    { 
        "row_conv_grad":
        [
            { "X":"x" },
            { "Filter":"filter" },
            { "Out":"out" },
        ]
    },
    { 
        "rsqrt":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "rsqrt_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "rsqrt_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "save_combine":
        [
            { "X":"x" },
        ]
    },
    { 
        "scale":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "scale_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "scatter":
        [
            { "X":"x" },
            { "Ids":"index" },
            { "Updates":"updates" },
            { "Out":"out" },
        ]
    },
    { 
        "scatter_grad":
        [
            { "X":"x" },
            { "Ids":"index" },
            { "Updates":"updates" },
            { "Out":"out" },
        ]
    },
    { 
        "scatter_nd_add":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Updates":"updates" },
            { "Out":"out" },
        ]
    },
    { 
        "scatter_nd_add_grad":
        [
            { "X":"x" },
            { "Index":"index" },
            { "Updates":"updates" },
            { "Out":"out" },
        ]
    },
    { 
        "searchsorted":
        [
            { "SortedSequence":"sorted_sequence" },
            { "Values":"values" },
            { "Out":"out" },
        ]
    },
    { 
        "seed":
        [
            { "Out":"out" },
        ]
    },
    { 
        "segment_pool":
        [
            { "X":"x" },
            { "SegmentIds":"segment_ids" },
            { "Out":"out" },
            { "SummedIds":"summed_ids" },
        ]
    },
    { 
        "segment_pool_grad":
        [
            { "X":"x" },
            { "SegmentIds":"segment_ids" },
            { "Out":"out" },
            { "SummedIds":"summed_ids" },
        ]
    },
    { 
        "self_dp_attention":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "selu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "selu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "graph_send_recv":
        [
            { "X":"x" },
            { "Src_index":"src_index" },
            { "Dst_index":"dst_index" },
            { "Out":"out" },
            { "Dst_count":"dst_count" },
        ]
    },
    { 
        "graph_send_recv_grad":
        [
            { "X":"x" },
            { "Src_index":"src_index" },
            { "Dst_index":"dst_index" },
            { "Out":"out" },
            { "Dst_count":"dst_count" },
        ]
    },
    { 
        "graph_send_ue_recv":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Src_index":"src_index" },
            { "Dst_index":"dst_index" },
            { "Out":"out" },
            { "Dst_count":"dst_count" },
        ]
    },
    { 
        "graph_send_ue_recv_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Src_index":"src_index" },
            { "Dst_index":"dst_index" },
            { "Out":"out" },
            { "Dst_count":"dst_count" },
        ]
    },
    { 
        "sequence_mask":
        [
            { "X":"x" },
            { "maxlen":"max_len" },
            { "Y":"y" },
        ]
    },
    { 
        "sgd":
        [
            { "Param":"param" },
            { "LearningRate":"learning_rate" },
            { "Grad":"grad" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "shape":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "shard_index":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "share_buffer":
        [
            { "X":"x" },
            { "Out":"out" },
            { "XOut":"xout" },
        ]
    },
    { 
        "shuffle_batch":
        [
            { "X":"x" },
            { "Seed":"seed" },
            { "Out":"out" },
            { "ShuffleIdx":"shuffle_idx" },
            { "SeedOut":"seed_out" },
        ]
    },
    { 
        "shuffle_batch_grad":
        [
            { "X":"x" },
            { "Seed":"seed" },
            { "Out":"out" },
            { "ShuffleIdx":"shuffle_idx" },
            { "SeedOut":"seed_out" },
        ]
    },
    { 
        "sigmoid":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sigmoid_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sigmoid_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sigmoid_triple_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sign":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sign_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "silu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "silu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "silu_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sin":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sin_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sin_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sin_triple_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sinh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sinh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "slice":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "slice_grad":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "slogdeterminant":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "slogdeterminant_grad":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "soft_relu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "soft_relu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softmax":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softmax_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softplus":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softplus_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softplus_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softshrink":
        [
            { "X":"x" },
            { "lambda":"threshold" },
            { "Out":"out" },
        ]
    },
    { 
        "softshrink_grad":
        [
            { "X":"x" },
            { "lambda":"threshold" },
            { "Out":"out" },
        ]
    },
    { 
        "softsign":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "softsign_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "solve":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "spectral_norm":
        [
            { "Weight":"weight" },
            { "U":"u" },
            { "V":"v" },
            { "Out":"out" },
        ]
    },
    { 
        "spectral_norm_grad":
        [
            { "Weight":"weight" },
            { "U":"u" },
            { "V":"v" },
            { "Out":"out" },
        ]
    },
    { 
        "split":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "split_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sqrt":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sqrt_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sqrt_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "square":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "square_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "square_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "squeeze2":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "squeeze2_grad":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "squeeze2_double_grad":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "stack":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "stack_grad":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "stanh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "stanh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "strided_slice":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "strided_slice_grad":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_sub":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "elementwise_sub_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_sum":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "out_dtype":"dtype" },
            { "Out":"out" },
        ]
    },
    { 
        "reduce_sum_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "out_dtype":"dtype" },
            { "Out":"out" },
        ]
    },
    { 
        "sum_double_grad":
        [
            { "X":"x" },
            { "dim":"axis" },
            { "keep_dim":"keepdim" },
            { "out_dtype":"dtype" },
            { "Out":"out" },
        ]
    },
    { 
        "svd":
        [
            { "X":"x" },
            { "U":"u" },
            { "S":"s" },
            { "VH":"vh" },
        ]
    },
    { 
        "svd_grad":
        [
            { "X":"x" },
            { "U":"u" },
            { "S":"s" },
            { "VH":"vh" },
        ]
    },
    { 
        "swish":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "swish_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sync_batch_norm":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "data_layout":"data_format" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "sync_batch_norm_grad":
        [
            { "X":"x" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Mean":"mean" },
            { "Variance":"variance" },
            { "data_layout":"data_format" },
            { "Y":"out" },
            { "MeanOut":"mean_out" },
            { "VarianceOut":"variance_out" },
            { "SavedMean":"saved_mean" },
            { "SavedVariance":"saved_variance" },
            { "ReserveSpace":"reserve_space" },
        ]
    },
    { 
        "take_along_axis":
        [
            { "Input":"arr" },
            { "Index":"indices" },
            { "Axis":"axis" },
            { "Result":"out" },
        ]
    },
    { 
        "take_along_axis_grad":
        [
            { "Input":"arr" },
            { "Index":"indices" },
            { "Axis":"axis" },
            { "Result":"out" },
        ]
    },
    { 
        "tan":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tan_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tanh":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tanh_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tanh_grad_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tanh_triple_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tanh_shrink":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tdm_sampler":
        [
            { "X":"x" },
            { "Travel":"travel" },
            { "Layer":"layer" },
            { "Out":"out" },
            { "Labels":"labels" },
            { "Mask":"mask" },
        ]
    },
    { 
        "thresholded_relu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tile":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tile_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tile_double_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "top_k_v2":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "top_k_v2_grad":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
        ]
    },
    { 
        "trace":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "transpose2":
        [
            { "X":"x" },
            { "axis":"perm" },
            { "Out":"out" },
        ]
    },
    { 
        "transpose2_grad":
        [
            { "X":"x" },
            { "axis":"perm" },
            { "Out":"out" },
        ]
    },
    { 
        "triangular_solve":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "triangular_solve_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "tril_triu":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "tril_triu_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "trilinear_interp_v2":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "trilinear_interp_v2_grad":
        [
            { "X":"x" },
            { "OutSize":"out_size" },
            { "SizeTensor":"size_tensor" },
            { "Scale":"scale_tensor" },
            { "data_layout":"data_format" },
            { "Out":"output" },
        ]
    },
    { 
        "trunc":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "truncated_gaussian_random":
        [
            { "Out":"out" },
        ]
    },
    { 
        "unbind":
        [
            { "X":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "unfold":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "uniform_random":
        [
            { "Out":"out" },
        ]
    },
    { 
        "uniform_random_inplace":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "uniform_random_inplace_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "unique":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Indices":"indices" },
            { "Index":"inverse" },
            { "Counts":"counts" },
        ]
    },
    { 
        "unique_consecutive":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Index":"index" },
            { "Counts":"counts" },
        ]
    },
    { 
        "unpool":
        [
            { "X":"x" },
            { "Indices":"indices" },
            { "paddings":"padding" },
            { "Out":"out" },
        ]
    },
    { 
        "unpool3d":
        [
            { "X":"x" },
            { "Indices":"indices" },
            { "Out":"out" },
        ]
    },
    { 
        "unsqueeze2":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "unsqueeze2_grad":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "unsqueeze2_double_grad":
        [
            { "X":"x" },
            { "axes":"axis" },
            { "Out":"out" },
            { "XShape":"xshape" },
        ]
    },
    { 
        "unstack":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "unstack_grad":
        [
            { "X":"x" },
            { "Y":"out" },
        ]
    },
    { 
        "update_loss_scaling":
        [
            { "X":"x" },
            { "FoundInfinite":"found_infinite" },
            { "PrevLossScaling":"prev_loss_scaling" },
            { "InGoodSteps":"in_good_steps" },
            { "InBadSteps":"in_bad_steps" },
            { "Out":"out" },
            { "LossScaling":"loss_scaling" },
            { "OutGoodSteps":"out_good_steps" },
            { "OutBadSteps":"out_bad_steps" },
        ]
    },
    { 
        "viterbi_decode":
        [
            { "Input":"potentials" },
            { "Transition":"transition_params" },
            { "Length":"lengths" },
            { "Scores":"scores" },
            { "Path":"path" },
        ]
    },
    { 
        "warpctc":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "LogitsLength":"logits_length" },
            { "LabelLength":"labels_length" },
            { "WarpCTCGrad":"warpctcgrad" },
            { "Loss":"loss" },
        ]
    },
    { 
        "warpctc_grad":
        [
            { "Logits":"logits" },
            { "Label":"label" },
            { "LogitsLength":"logits_length" },
            { "LabelLength":"labels_length" },
            { "WarpCTCGrad":"warpctcgrad" },
            { "Loss":"loss" },
        ]
    },
    { 
        "where":
        [
            { "Condition":"condition" },
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "where_grad":
        [
            { "Condition":"condition" },
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
    { 
        "yolo_box":
        [
            { "X":"x" },
            { "ImgSize":"img_size" },
            { "Boxes":"boxes" },
            { "Scores":"scores" },
        ]
    },
    { 
        "yolov3_loss":
        [
            { "X":"x" },
            { "GTBox":"gt_box" },
            { "GTLabel":"gt_label" },
            { "GTScore":"gt_score" },
            { "Loss":"loss" },
            { "ObjectnessMask":"objectness_mask" },
            { "GTMatchMask":"gt_match_mask" },
        ]
    },
    { 
        "yolov3_loss_grad":
        [
            { "X":"x" },
            { "GTBox":"gt_box" },
            { "GTLabel":"gt_label" },
            { "GTScore":"gt_score" },
            { "Loss":"loss" },
            { "ObjectnessMask":"objectness_mask" },
            { "GTMatchMask":"gt_match_mask" },
        ]
    },
    { 
        "c_allgather":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_allreduce_max":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_allreduce_min":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_allreduce_prod":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_allreduce_sum":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_broadcast":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_identity":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_reduce_min":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_reduce_sum":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_reducescatter":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_sync_calc_stream":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "c_sync_comm_stream":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "channel_shuffle":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "decayed_adagrad":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Moment":"moment" },
            { "LearningRate":"learning_rate" },
            { "ParamOut":"param_out" },
            { "MomentOut":"moment_out" },
        ]
    },
    { 
        "distribute_fpn_proposals":
        [
            { "FpnRois":"fpn_rois" },
            { "RoisNum":"rois_num" },
            { "MultiFpnRois":"multi_fpn_rois" },
            { "MultiLevelRoIsNum":"multi_level_rois_num" },
            { "RestoreIndex":"restore_index" },
        ]
    },
    { 
        "distributed_lookup_table":
        [
            { "Ids":"ids" },
            { "W":"w" },
            { "Outputs":"outputs" },
        ]
    },
    { 
        "dpsgd":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "ParamOut":"param_out" },
        ]
    },
    { 
        "fetch_v2":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "ftrl":
        [
            { "Param":"param" },
            { "SquaredAccumulator":"squared_accumulator" },
            { "LinearAccumulator":"linear_accumulator" },
            { "Grad":"grad" },
            { "LearningRate":"learning_rate" },
            { "ParamOut":"param_out" },
            { "SquaredAccumOut":"squared_accum_out" },
            { "LinearAccumOut":"linear_accum_out" },
        ]
    },
    { 
        "fill_constant_batch_size_like":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "fused_elemwise_add_activation":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
            { "IntermediateOut":"intermediate_out" },
        ]
    },
    { 
        "fused_elemwise_add_activation_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
            { "IntermediateOut":"intermediate_out" },
        ]
    },
    { 
        "fusion_squared_mat_sub":
        [
            { "X":"x" },
            { "Y":"y" },
            { "SquaredX":"squared_x" },
            { "SquaredY":"squared_y" },
            { "SquaredXY":"squared_xy" },
            { "Out":"out" },
        ]
    },
    { 
        "get_tensor_from_selected_rows":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "identity_loss":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "lars_momentum":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Velocity":"velocity" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "VelocityOut":"velocity_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "lod_array_length":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "logspace":
        [
            { "Start":"start" },
            { "Stop":"stop" },
            { "Num":"num" },
            { "Base":"base" },
            { "Out":"out" },
        ]
    },
    { 
        "lu":
        [
            { "X":"x" },
            { "pivots":"pivot" },
            { "Out":"out" },
            { "Pivots":"pivots" },
            { "Infos":"infos" },
        ]
    },
    { 
        "lu_grad":
        [
            { "X":"x" },
            { "pivots":"pivot" },
            { "Out":"out" },
            { "Pivots":"pivots" },
            { "Infos":"infos" },
        ]
    },
    { 
        "match_matrix_tensor":
        [
            { "X":"x" },
            { "Y":"y" },
            { "W":"w" },
            { "Out":"out" },
            { "Tmp":"tmp" },
        ]
    },
    { 
        "match_matrix_tensor_grad":
        [
            { "X":"x" },
            { "Y":"y" },
            { "W":"w" },
            { "Out":"out" },
            { "Tmp":"tmp" },
        ]
    },
    { 
        "memcpy":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "memcpy_d2h":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "nce":
        [
            { "Input":"input" },
            { "Label":"label" },
            { "Weight":"weight" },
            { "Bias":"bias" },
            { "SampleWeight":"sample_weight" },
            { "CustomDistProbs":"custom_dist_probs" },
            { "CustomDistAlias":"custom_dist_alias" },
            { "CustomDistAliasProbs":"custom_dist_alias_probs" },
            { "Cost":"cost" },
            { "SampleLogits":"sample_logits" },
            { "SampleLabels":"sample_labels" },
        ]
    },
    { 
        "nce_grad":
        [
            { "Input":"input" },
            { "Label":"label" },
            { "Weight":"weight" },
            { "Bias":"bias" },
            { "SampleWeight":"sample_weight" },
            { "CustomDistProbs":"custom_dist_probs" },
            { "CustomDistAlias":"custom_dist_alias" },
            { "CustomDistAliasProbs":"custom_dist_alias_probs" },
            { "Cost":"cost" },
            { "SampleLogits":"sample_logits" },
            { "SampleLabels":"sample_labels" },
        ]
    },
    { 
        "number_count":
        [
            { "numbers":"numbers" },
            { "Out":"out" },
        ]
    },
    { 
        "partial_send":
        [
            { "X":"x" },
        ]
    },
    { 
        "read_from_array":
        [
            { "X":"array" },
            { "I":"i" },
            { "Out":"out" },
        ]
    },
    { 
        "recv_v2":
        [
            { "Out":"out" },
        ]
    },
    { 
        "graph_reindex":
        [
            { "X":"x" },
            { "Neighbors":"neighbors" },
            { "Count":"count" },
            { "HashTable_Value":"hashtable_value" },
            { "HashTable_Index":"hashtable_index" },
            { "Reindex_Src":"reindex_src" },
            { "Reindex_Dst":"reindex_dst" },
            { "Out_Nodes":"out_nodes" },
        ]
    },
    { 
        "rrelu":
        [
            { "X":"x" },
            { "Out":"out" },
            { "Noise":"noise" },
        ]
    },
    { 
        "send_v2":
        [
            { "X":"x" },
        ]
    },
    { 
        "set_value":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "set_value_grad":
        [
            { "Input":"x" },
            { "Out":"out" },
            { "ValueTensor@GRAD":"values_grad" },
        ]
    },
    { 
        "set_value_with_tensor":
        [
            { "Input":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "share_data":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "sigmoid_cross_entropy_with_logits":
        [
            { "X":"x" },
            { "Label":"label" },
            { "Out":"out" },
        ]
    },
    { 
        "sigmoid_cross_entropy_with_logits_grad":
        [
            { "X":"x" },
            { "Label":"label" },
            { "Out":"out" },
        ]
    },
    { 
        "skip_layernorm":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Scale":"scale" },
            { "Bias":"bias" },
            { "Out":"out" },
        ]
    },
    { 
        "sparse_momentum":
        [
            { "Param":"param" },
            { "Grad":"grad" },
            { "Velocity":"velocity" },
            { "Index":"index" },
            { "Axis":"axis" },
            { "LearningRate":"learning_rate" },
            { "MasterParam":"master_param" },
            { "ParamOut":"param_out" },
            { "VelocityOut":"velocity_out" },
            { "MasterParamOut":"master_param_out" },
        ]
    },
    { 
        "squared_l2_norm":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "squared_l2_norm_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "temporal_shift":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "temporal_shift_grad":
        [
            { "X":"x" },
            { "Out":"out" },
        ]
    },
    { 
        "uniform_random_batch_size_like":
        [
            { "Input":"input" },
            { "Out":"out" },
        ]
    },
    { 
        "write_to_array":
        [
            { "X":"x" },
            { "I":"i" },
            { "Out":"out" },
        ]
    },
    { 
        "deformable_conv_v1":
        [
            { "Input":"x" },
            { "Offset":"offset" },
            { "Filter":"filter" },
            { "Mask":"mask" },
            { "Output":"out" },
        ]
    },
    { 
        "fetch":
        [
            { "X":"x" },
        ]
    },
    { 
        "matmul":
        [
            { "X":"x" },
            { "Y":"y" },
            { "Out":"out" },
        ]
    },
]
    