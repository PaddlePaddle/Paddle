#pragma once
/*
 * ATTENTION this header file can only include in .cc file.
 */

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(relu);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);
USE_LITE_OP(elementwise_add)
USE_LITE_OP(elementwise_sub)
USE_LITE_OP(square)
USE_LITE_OP(softmax)
USE_LITE_OP(dropout)
USE_LITE_OP(concat)
USE_LITE_OP(conv2d)
USE_LITE_OP(depthwise_conv2d)
USE_LITE_OP(pool2d)
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
