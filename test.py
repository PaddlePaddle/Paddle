import paddle.fluid as fluid
def create_offset(x):
    trans = fluid.layers.fill_constant(shape=[1, 32, 8, 8],
                                       dtype='float32', 
                                       value=0.0)
    trans = fluid.layers.nn.conv2d(input=x, num_filters=2, 
                                   filter_size=1, name='conv2f_2', 
                                   act="sigmoid")
    trans = fluid.layers.nn.deformable_psroi_pooling(input=x, 
                                                     rois=bbox, 
                                                     trans=trans, 
                                                     no_trans=1,
                                                     spatial_scale=1.0, 
                                                     output_dim=3,                                                             
                                                     group_size=1, 
                                                     pooled_size=8, 
                                                     part_size=8, 
                                                     sample_per_part=4, 
                                                     trans_std=0.1)
    trans = fluid.layers.nn.conv2d(input=trans, 
                                   num_filters=2,
                                   filter_size=1, 
                                   stride = 1, 
                                   act='sigmoid')
    return trans


    input = fluid.layers.data(name="input",
                              shape=[2, 3, 64, 64], 
                              dtype='float32', 
                              append_batch_size=False)
                     
    bbox = fluid.layers.data(name="bbox",shape=[4],
                             dtype='float32', lod_level=1)
            
    x = fluid.layers.nn.conv2d(input=input, num_filters=3, 
                               filter_size=3, stride = 2,  
                               padding=1,name='conv2d_1')
    x = create_offset(x)            
    x = fluid.layers.nn.deformable_psroi_pooling(input=x, 
                                                 rois=bbox, 
                                                 trans=trans, 
                                                 no_trans=0,
                                                 spatial_scale=1.0, 
                                                  output_dim=3,
                                                  group_size=1,
                                                  pooled_size=8, 
                                                  part_size=8, 
                                                  sample_per_part=4, 
                                                  trans_std=0.1)

