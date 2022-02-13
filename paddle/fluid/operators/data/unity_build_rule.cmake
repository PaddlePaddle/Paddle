# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc
    pipeline.cc
    map_runner.cc
    random_roi_generator.cc
    nvjpeg_decoder.cc
    dataloader_op.cc
    map_op.cc
    batch_decode_random_crop_op.cc
    random_flip_op.cc)

register_unity_group(cu
    dataloader_op.cu.cc
    map_op.cu.cc
    batch_decode_random_crop_op.cu)
