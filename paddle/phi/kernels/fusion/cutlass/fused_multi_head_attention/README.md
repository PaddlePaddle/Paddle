The fused_multi_head_attention kernel is modified from [xformers's implementation](https://github.com/facebookresearch/xformers/tree/main/xformers/csrc/attention/cuda/fmha)

Now we only implement forward kernels, and the templates are: 

- T: Means Dtype, currently we only register fp16. 
- ArchTag: Means SM Arch, currently we only register SM70 SM75 SM80. 
- IsAligned: Whether the data pointer is aligned with 128 bit and the last dim is aligned with 16 / sizeof(T) so that we can pack. Currently we assume all input is aligned. 
- QueriesPerBlock: Each block process the number of Query seq_len. 
- KeysPerBlock: Each block process the number of Key seq_len. 
- AddMask: Whether the kernel need to add mask. Since we found here is performance loss if we pass argument instead of template. 
- MaskBroadcastRow: Whether the mask is broadcast at row, its shape like (1, seq_len). 

We use `generate_kernels.py` to generate kernels with different configurations. 

The kernels is generated in `kernels/impl` directory, and its filename is like `cutlass_fmha_forward_{DTYPE}_{ALIGNED}_{SM_ARCH}.cu`. 

If you want to add new configurations like add fp32 type, you should modify `generate_kernels.py` first and then run it to generate kernels.
