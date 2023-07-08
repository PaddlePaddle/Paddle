<!-- Demo: https://github.com/PaddlePaddle/Paddle/pull/24810 -->
### PR types
New Features 

### PR changes
OPs 

### Description
Implemented the following functions in the Paddle API:

`affine_grid`: Generates a grid of coordinates based on a transformation matrix.
`pixel_shuffle`: Rearranges the elements of a tensor to perform pixel shuffling.
`pixel_unshuffle`: Reverses the operation of pixel shuffling.
`grid_sample`: Performs bilinear interpolation on a grid of coordinates.
`channel_shuffle`: Rearranges the channels of a tensor.

Each function is accompanied by its respective unit tests to ensure correctness.
Added the functions to the `paddle.api.h` file.