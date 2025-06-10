import paddle
from paddle import _C_ops, in_dynamic_mode


def fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs, name=None):
    r"""
    Fused SwiGLU probability gradient computation for efficient MoE (Mixture of Experts) training.
    
    This operator computes the backward pass of SwiGLU activation with probability weighting:
    
    Forward: o2 = SiLU(x1) * x2 * prob
    where SiLU(x) = x * sigmoid(x)
    
    Args:
        o1 (Tensor): Input tensor containing concatenated gate and up projections.
                     Shape: [..., hidden_size * 2], dtype: bfloat16
        do2_s (Tensor): Gradient of the scaled output tensor.
                        Shape: [..., hidden_size], dtype: bfloat16  
        unzipped_probs (Tensor): Probability weights for each sample.
                                 Shape: [...], dtype: float32
                                 Must have same batch dimensions as o1 and do2_s
        name (str, optional): The default value is None. Normally there is no need for user 
                              to set this property. For more information, please refer to 
                              :ref:`api_guide_Name`.
    
    Returns:
        tuple: A tuple containing three tensors:
        
        - **do1** (Tensor): Gradient w.r.t. input o1. Shape: [..., hidden_size * 2], dtype: bfloat16
        - **probs_grad** (Tensor): Gradient w.r.t. probabilities. Shape: [...], dtype: float32  
        - **o2_s** (Tensor): Scaled output o2 * prob. Shape: [..., hidden_size], dtype: bfloat16
    
    Examples:
        .. code-block:: python
        
            import paddle
            from paddle.incubate.nn.functional import fused_swiglu_probs_grad
            
            # Example 1: Basic 2D usage
            batch_size, hidden_size = 8, 2048
            o1 = paddle.randn([batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([batch_size], dtype='float32')
            
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
            
            # Example 2: 3D tensor (sequence + batch)
            seq_len, batch_size, hidden_size = 512, 8, 2048
            o1 = paddle.randn([seq_len, batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([seq_len, batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([seq_len, batch_size], dtype='float32')
            
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
            
            # Example 3: MoE scenario with 4D tensors
            seq_len, top_k, batch_size, hidden_size = 512, 2, 8, 2048
            o1 = paddle.randn([seq_len, top_k, batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([seq_len, top_k, batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([seq_len, top_k, batch_size], dtype='float32')
            
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
    
    Note:
        - This operator is specifically optimized for MoE training scenarios
        - All input tensors must be on the same device (GPU)
        - The operator leverages vectorized CUDA kernels for optimal performance
        - Batch dimensions (all dimensions except the last for o1/do2_s) must match across inputs
    """
    
    # 参数验证
    if not isinstance(o1, paddle.Tensor):
        raise TypeError(f"o1 must be a Tensor, but got {type(o1)}")
    if not isinstance(do2_s, paddle.Tensor):
        raise TypeError(f"do2_s must be a Tensor, but got {type(do2_s)}")
    if not isinstance(unzipped_probs, paddle.Tensor):
        raise TypeError(f"unzipped_probs must be a Tensor, but got {type(unzipped_probs)}")
    
    # 数据类型验证
    if o1.dtype != paddle.bfloat16:
        raise ValueError(f"o1 must have dtype bfloat16, but got {o1.dtype}")
    if do2_s.dtype != paddle.bfloat16:
        raise ValueError(f"do2_s must have dtype bfloat16, but got {do2_s.dtype}")
    if unzipped_probs.dtype != paddle.float32:
        raise ValueError(f"unzipped_probs must have dtype float32, but got {unzipped_probs.dtype}")
    
    # 设备验证
    if o1.place != do2_s.place or o1.place != unzipped_probs.place:
        raise ValueError("All input tensors must be on the same device")
    
    # 基本维度验证
    if len(o1.shape) != len(do2_s.shape):
        raise ValueError(f"o1 and do2_s must have same number of dimensions, "
                        f"but got {len(o1.shape)} vs {len(do2_s.shape)}")
    
    if o1.shape[-1] != do2_s.shape[-1] * 2:
        raise ValueError(f"Last dimension of o1 must be twice that of do2_s, "
                        f"but got {o1.shape[-1]} vs {do2_s.shape[-1] * 2}")
    
    # 调用底层算子
    if in_dynamic_mode():
        return _C_ops.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)
    else:
        # 静态图模式
        from paddle.static import default_main_program
        helper = paddle.static.LayerHelper('fused_swiglu_weighted_bwd', **locals())
        
        # 创建输出变量
        do1 = helper.create_variable_for_type_inference(dtype=o1.dtype)
        probs_grad = helper.create_variable_for_type_inference(dtype=paddle.float32)
        o2_s = helper.create_variable_for_type_inference(dtype=do2_s.dtype)
        
        helper.append_op(
            type='fused_swiglu_weighted_bwd',
            inputs={
                'o1': o1,
                'do2_s': do2_s, 
                'unzipped_probs': unzipped_probs
            },
            outputs={
                'do1': do1,
                'probs_grad': probs_grad,
                'o2_s': o2_s
            },
            attrs={}
        )
        
        return do1, probs_grad, o2_s