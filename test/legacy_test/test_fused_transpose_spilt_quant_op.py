

import numpy as np
import paddle
import paddle.nn.functional as F


def restore_transpose_split_quant(outs, scales):
    """恢复原始数据用于验证"""
 
    outs_float32 = [out.astype('float32') for out in outs]
    

    concatenated_out = paddle.concat(outs_float32, axis=1)  # [K, total_tokens]
    transposed_out = concatenated_out.transpose([1, 0])     # [total_tokens, K]
    

    concatenated_scale = paddle.concat(scales, axis=0)      # [total_groups, K]
    expanded_scale = paddle.repeat_interleave(
        concatenated_scale, repeats=128, axis=0
    )  # [total_tokens, K]
    

    return transposed_out * expanded_scale


def test_fused_transpose_split_quant(tokens_per_expert, seq_len, pow_2_scales):
    
    print(f"Testing: tokens_per_expert={tokens_per_expert}, seq_len={seq_len}, pow_2_scales={pow_2_scales}")
    

    valid_tokens = [t for t in tokens_per_expert if t > 0]
    
    if not valid_tokens or seq_len == 0:
        print("  → Skipping empty case")
        return
    
  
    x = paddle.randn([sum(valid_tokens), seq_len], dtype='bfloat16')
    x = paddle.clip(x, min=-50, max=50)
    
    try:

        outs, scales = F.fused_transpose_split_quant(
            x, valid_tokens, pow_2_scales=pow_2_scales
        )
        
   
        assert len(outs) == len(valid_tokens), f"Expected {len(valid_tokens)} outputs, got {len(outs)}"
        assert len(scales) == len(valid_tokens), f"Expected {len(valid_tokens)} scales, got {len(scales)}"
        

        for i, tokens in enumerate(valid_tokens):
            expected_out_shape = [seq_len, tokens]
            expected_scale_shape = [tokens // 128, seq_len]
            
            assert list(outs[i].shape) == expected_out_shape, \
                f"Output {i} shape mismatch: expected {expected_out_shape}, got {list(outs[i].shape)}"
            assert list(scales[i].shape) == expected_scale_shape, \
                f"Scale {i} shape mismatch: expected {expected_scale_shape}, got {list(scales[i].shape)}"
            

            assert outs[i].dtype == paddle.float8_e4m3fn, f"Output {i} dtype should be float8_e4m3fn"
            assert scales[i].dtype == paddle.float32, f"Scale {i} dtype should be float32"
        

        x_restored = restore_transpose_split_quant(outs, scales)
        x_float32 = x.astype('float32')
        
        np.testing.assert_allclose(
            x_float32.numpy(), x_restored.numpy(), 
            rtol=0.02, atol=0.5,
            err_msg=f"Numerical accuracy test failed for tokens={valid_tokens}, seq_len={seq_len}"
        )
        
        print("  ✓ PASSED")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        raise


def run_all_tests():
    print("=" * 60)
    print("Testing fused_transpose_split_quant with new API")
    print("=" * 60)
    
    paddle.seed(42)
    np.random.seed(42)
    

    if paddle.is_compiled_with_cuda():
        paddle.device.set_device('gpu:0')
        print("Using GPU for testing")
    else:
        paddle.device.set_device('cpu')
        print("Using CPU for testing")
    
    test_cases = [
 
        ([128, 256], 1024, False),
        ([128, 256], 1024, True),
        

        ([128], 1, False),
        ([128], 1, True),
        

        ([3*128, 4*128, 5*128], 233, False),
        ([3*128, 4*128, 5*128], 233, True),
        
   
        ([24*128, 128, 50*128, 16*128], 2162, True),
        ([7*128, 29*128, 3*128, 128*128, 13*128], 4000, False),
        
 
        ([18*128, 5*128, 24*128, 128, 6*128, 27*128, 7*128], 7168, True),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for tokens_per_expert, seq_len, pow_2_scales in test_cases:
        try:
            test_fused_transpose_split_quant(tokens_per_expert, seq_len, pow_2_scales)
            success_count += 1
        except Exception as e:
            print(f"Test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_count} tests passed")
    if success_count == total_count:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {total_count - success_count} tests failed")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()