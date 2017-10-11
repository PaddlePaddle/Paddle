..  _api_v2.layer:

======
Layers
======

Data layer
===========

..  _api_v2.layer_data:

data
----
..  autoclass:: paddle.v2.layer.data
    :noindex:

Fully Connected Layers
======================

..  _api_v2.layer_fc:

fc
--
..  autoclass:: paddle.v2.layer.fc
    :noindex:

selective_fc
------------
..  autoclass:: paddle.v2.layer.selective_fc
    :noindex:

Conv Layers
===========

conv_operator
-------------
..  autoclass:: paddle.v2.layer.conv_operator
    :noindex:

conv_projection
---------------
..  autoclass:: paddle.v2.layer.conv_projection
    :noindex:

conv_shift
----------
..  autoclass:: paddle.v2.layer.conv_shift
    :noindex:

img_conv
--------
..  autoclass:: paddle.v2.layer.img_conv
    :noindex:

..  _api_v2.layer_context_projection:

context_projection 
------------------
..  autoclass:: paddle.v2.layer.context_projection
    :noindex:

Image Pooling Layer
===================

img_pool
--------
..  autoclass:: paddle.v2.layer.img_pool
    :noindex:   

spp
---
..  autoclass:: paddle.v2.layer.spp
    :noindex:

maxout
------
..  autoclass:: paddle.v2.layer.maxout
    :noindex:

Norm Layer
==========

img_cmrnorm
-----------
..  autoclass:: paddle.v2.layer.img_cmrnorm
    :noindex:

batch_norm
----------
..  autoclass:: paddle.v2.layer.batch_norm
    :noindex:

sum_to_one_norm
---------------
..  autoclass:: paddle.v2.layer.sum_to_one_norm
    :noindex:
    
cross_channel_norm
------------------
..  autoclass:: paddle.v2.layer.cross_channel_norm
    :noindex:
    
Recurrent Layers
================

recurrent
---------
..  autoclass:: paddle.v2.layer.recurrent
    :noindex:

lstmemory
---------
..  autoclass:: paddle.v2.layer.lstmemory
    :noindex:

grumemory
---------
..  autoclass:: paddle.v2.layer.grumemory
    :noindex:

Recurrent Layer Group
=====================

memory
------
..  autoclass:: paddle.v2.layer.memory
    :noindex:

recurrent_group
---------------
..  autoclass:: paddle.v2.layer.recurrent_group
    :noindex:
    
lstm_step
---------
..  autoclass:: paddle.v2.layer.lstm_step
    :noindex:

gru_step
--------
..  autoclass:: paddle.v2.layer.gru_step
    :noindex:

beam_search
------------
..  autoclass:: paddle.v2.layer.beam_search
    :noindex:
    
get_output
----------
..  autoclass:: paddle.v2.layer.get_output
    :noindex:
    
Mixed Layer
===========

..  _api_v2.layer_mixed:

mixed
-----
..  autoclass:: paddle.v2.layer.mixed
    :noindex:

..  _api_v2.layer_embedding:

embedding
---------
..  autoclass:: paddle.v2.layer.embedding
    :noindex:

scaling_projection
------------------
..  autoclass:: paddle.v2.layer.scaling_projection
    :noindex:

dotmul_projection
-----------------
..  autoclass:: paddle.v2.layer.dotmul_projection
    :noindex:

dotmul_operator
---------------
..  autoclass:: paddle.v2.layer.dotmul_operator
    :noindex:

full_matrix_projection
----------------------
..  autoclass:: paddle.v2.layer.full_matrix_projection
    :noindex:

identity_projection
-------------------
..  autoclass:: paddle.v2.layer.identity_projection
    :noindex:


table_projection
----------------
..  autoclass:: paddle.v2.layer.table_projection
    :noindex:

trans_full_matrix_projection
----------------------------
..  autoclass:: paddle.v2.layer.trans_full_matrix_projection
    :noindex:
    
Aggregate Layers
================

..  _api_v2.layer_pooling:

pooling
-------
..  autoclass:: paddle.v2.layer.pooling
    :noindex:

..  _api_v2.layer_last_seq:

last_seq
--------
..  autoclass:: paddle.v2.layer.last_seq
    :noindex:

..  _api_v2.layer_first_seq:

first_seq
---------
..  autoclass:: paddle.v2.layer.first_seq
    :noindex:

concat
------
..  autoclass:: paddle.v2.layer.concat
    :noindex:

seq_concat
----------
..  autoclass:: paddle.v2.layer.seq_concat
    :noindex:

Reshaping Layers
================

block_expand
------------
..  autoclass:: paddle.v2.layer.block_expand
    :noindex:

..  _api_v2.layer_expand:

expand
------
..  autoclass:: paddle.v2.layer.expand
    :noindex:

repeat
------
..  autoclass:: paddle.v2.layer.repeat
    :noindex:

rotate
------
..  autoclass:: paddle.v2.layer.rotate
    :noindex:

seq_reshape
-----------
..  autoclass:: paddle.v2.layer.seq_reshape
    :noindex:

Math Layers
===========

addto
-----
..  autoclass:: paddle.v2.layer.addto
    :noindex:

linear_comb
-----------
..  autoclass:: paddle.v2.layer.linear_comb
    :noindex:

interpolation
-------------
..  autoclass:: paddle.v2.layer.interpolation
    :noindex:

bilinear_interp
---------------
..  autoclass:: paddle.v2.layer.bilinear_interp
    :noindex:

power
-----
..  autoclass:: paddle.v2.layer.power
    :noindex:

scaling
-------
..  autoclass:: paddle.v2.layer.scaling
    :noindex:

slope_intercept
---------------
..  autoclass:: paddle.v2.layer.slope_intercept
    :noindex:

tensor
------
..  autoclass:: paddle.v2.layer.tensor
    :noindex:

..  _api_v2.layer_cos_sim:

cos_sim
-------
..  autoclass:: paddle.v2.layer.cos_sim
    :noindex:

trans
-----
..  autoclass:: paddle.v2.layer.trans
    :noindex:

Sampling Layers
===============

maxid
-----
..  autoclass:: paddle.v2.layer.max_id
    :noindex:

sampling_id
-----------
..  autoclass:: paddle.v2.layer.sampling_id
    :noindex:

Slicing and Joining Layers
==========================

pad
----
..  autoclass:: paddle.v2.layer.pad
    :noindex:

..  _api_v2.layer_costs:

Cost Layers
===========

cross_entropy_cost
------------------
..  autoclass:: paddle.v2.layer.cross_entropy_cost
    :noindex:

cross_entropy_with_selfnorm_cost
--------------------------------
..  autoclass:: paddle.v2.layer.cross_entropy_with_selfnorm_cost
    :noindex:

multi_binary_label_cross_entropy_cost
-------------------------------------
..  autoclass:: paddle.v2.layer.multi_binary_label_cross_entropy_cost
    :noindex:

huber_cost
----------
..  autoclass:: paddle.v2.layer.huber_cost
    :noindex:

lambda_cost
-----------
..  autoclass:: paddle.v2.layer.lambda_cost
    :noindex:

mse_cost
--------
..  autoclass:: paddle.v2.layer.mse_cost
    :noindex:

rank_cost
---------
..  autoclass:: paddle.v2.layer.rank_cost
    :noindex:

sum_cost
---------
..  autoclass:: paddle.v2.layer.sum_cost
    :noindex:

crf
---
..  autoclass:: paddle.v2.layer.crf
    :noindex:

crf_decoding
------------
..  autoclass:: paddle.v2.layer.crf_decoding
    :noindex:

ctc
---
..  autoclass:: paddle.v2.layer.ctc
    :noindex:

warp_ctc
--------
..  autoclass:: paddle.v2.layer.warp_ctc
    :noindex:

nce
---
..  autoclass:: paddle.v2.layer.nce
    :noindex:

hsigmoid
---------
..  autoclass:: paddle.v2.layer.hsigmoid
    :noindex:

Check Layer 
============

eos
---
..  autoclass:: paddle.v2.layer.eos
    :noindex:
