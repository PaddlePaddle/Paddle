..  _api_v2.layer:

======
Layers
======

Data layer
===========

..  _api_v2.layer_data:

data
----
..  automodule:: paddle.v2.layer
    :members: data
    :noindex:

Fully Connected Layers
======================

..  _api_v2.layer_fc:

fc
--
..  automodule:: paddle.v2.layer
    :members: fc
    :noindex:

selective_fc
------------
..  automodule:: paddle.v2.layer
    :members: selective_fc
    :noindex:

Conv Layers
===========

conv_operator
-------------
..  automodule:: paddle.v2.layer
    :members: conv_operator
    :noindex:

conv_projection
---------------
..  automodule:: paddle.v2.layer
    :members: conv_projection
    :noindex:

conv_shift
----------
..  automodule:: paddle.v2.layer
    :members: conv_shift
    :noindex:

img_conv
--------
..  automodule:: paddle.v2.layer
    :members: img_conv
    :noindex:

..  _api_v2.layer_context_projection:

context_projection 
------------------
..  automodule:: paddle.v2.layer
    :members: context_projection
    :noindex:

Image Pooling Layer
===================

img_pool
--------
..  automodule:: paddle.v2.layer
    :members: img_pool
    :noindex:   

spp
---
..  automodule:: paddle.v2.layer
    :members: spp
    :noindex:

maxout
------
..  automodule:: paddle.v2.layer
    :members: maxout
    :noindex:

Norm Layer
==========

img_cmrnorm
-----------
..  automodule:: paddle.v2.layer
    :members: img_cmrnorm
    :noindex:

batch_norm
----------
..  automodule:: paddle.v2.layer
    :members: batch_norm
    :noindex:

sum_to_one_norm
---------------
..  automodule:: paddle.v2.layer
    :members: sum_to_one_norm
    :noindex:
    
cross_channel_norm
------------------
..  automodule:: paddle.v2.layer
    :members: cross_channel_norm
    :noindex:
    
Recurrent Layers
================

recurrent
---------
..  automodule:: paddle.v2.layer
    :members: recurrent
    :noindex:

lstmemory
---------
..  automodule:: paddle.v2.layer
    :members: lstmemory
    :noindex:

grumemory
---------
..  automodule:: paddle.v2.layer
    :members: grumemory
    :noindex:

Recurrent Layer Group
=====================

memory
------
..  automodule:: paddle.v2.layer
    :members: memory
    :noindex:

recurrent_group
---------------
..  automodule:: paddle.v2.layer
    :members: recurrent_group
    :noindex:
    
lstm_step
---------
..  automodule:: paddle.v2.layer
    :members: lstm_step
    :noindex:

gru_step
--------
..  automodule:: paddle.v2.layer
    :members: gru_step
    :noindex:

beam_search
------------
..  automodule:: paddle.v2.layer
    :members: beam_search
    :noindex:
    
get_output
----------
..  automodule:: paddle.v2.layer
    :members: get_output
    :noindex:
    
Mixed Layer
===========

..  _api_v2.layer_mixed:

mixed
-----
..  automodule:: paddle.v2.layer
    :members: mixed
    :noindex:

..  _api_v2.layer_embedding:

embedding
---------
..  automodule:: paddle.v2.layer
    :members: embedding
    :noindex:

scaling_projection
------------------
..  automodule:: paddle.v2.layer
    :members: scaling_projection
    :noindex:

dotmul_projection
-----------------
..  automodule:: paddle.v2.layer
    :members: dotmul_projection
    :noindex:

dotmul_operator
---------------
..  automodule:: paddle.v2.layer
    :members: dotmul_operator
    :noindex:

full_matrix_projection
----------------------
..  automodule:: paddle.v2.layer
    :members: full_matrix_projection
    :noindex:

identity_projection
-------------------
..  automodule:: paddle.v2.layer
    :members: identity_projection
    :noindex:


table_projection
----------------
..  automodule:: paddle.v2.layer
    :members: table_projection
    :noindex:

trans_full_matrix_projection
----------------------------
..  automodule:: paddle.v2.layer
    :members: trans_full_matrix_projection
    :noindex:
    
Aggregate Layers
================

..  _api_v2.layer_pooling:

pooling
-------
..  automodule:: paddle.v2.layer
    :members: pooling
    :noindex:

..  _api_v2.layer_last_seq:

last_seq
--------
..  automodule:: paddle.v2.layer
    :members: last_seq
    :noindex:

..  _api_v2.layer_first_seq:

first_seq
---------
..  automodule:: paddle.v2.layer
    :members: first_seq
    :noindex:

concat
------
..  automodule:: paddle.v2.layer
    :members: concat
    :noindex:

seq_concat
----------
..  automodule:: paddle.v2.layer
    :members: seq_concat
    :noindex:

Reshaping Layers
================

block_expand
------------
..  automodule:: paddle.v2.layer
    :members: block_expand
    :noindex:

..  _api_v2.layer_expand:

expand
------
..  automodule:: paddle.v2.layer
    :members: expand
    :noindex:

repeat
------
..  automodule:: paddle.v2.layer
    :members: repeat
    :noindex:

rotate
------
..  automodule:: paddle.v2.layer
    :members: rotate
    :noindex:

seq_reshape
-----------
..  automodule:: paddle.v2.layer
    :members: seq_reshape
    :noindex:

Math Layers
===========

addto
-----
..  automodule:: paddle.v2.layer
    :members: addto
    :noindex:

linear_comb
-----------
..  automodule:: paddle.v2.layer
    :members: linear_comb
    :noindex:

interpolation
-------------
..  automodule:: paddle.v2.layer
    :members: interpolation
    :noindex:

bilinear_interp
---------------
..  automodule:: paddle.v2.layer
    :members: bilinear_interp
    :noindex:

power
-----
..  automodule:: paddle.v2.layer
    :members: power
    :noindex:

scaling
-------
..  automodule:: paddle.v2.layer
    :members: scaling
    :noindex:

slope_intercept
---------------
..  automodule:: paddle.v2.layer
    :members: slope_intercept
    :noindex:

tensor
------
..  automodule:: paddle.v2.layer
    :members: tensor
    :noindex:

..  _api_v2.layer_cos_sim:

cos_sim
-------
..  automodule:: paddle.v2.layer
    :members: cos_sim
    :noindex:

trans
-----
..  automodule:: paddle.v2.layer
    :members: trans
    :noindex:

Sampling Layers
===============

maxid
-----
..  automodule:: paddle.v2.layer
    :members: maxid
    :noindex:

sampling_id
-----------
..  automodule:: paddle.v2.layer
    :members: sampling_id
    :noindex:

Slicing and Joining Layers
==========================

pad
----
..  automodule:: paddle.v2.layer
    :members: pad
    :noindex:

..  _api_v2.layer_costs:

Cost Layers
===========

cross_entropy_cost
------------------
..  automodule:: paddle.v2.layer
    :members: cross_entropy_cost
    :noindex:

cross_entropy_with_selfnorm_cost
--------------------------------
..  automodule:: paddle.v2.layer
    :members: cross_entropy_with_selfnorm_cost
    :noindex:

multi_binary_label_cross_entropy_cost
-------------------------------------
..  automodule:: paddle.v2.layer
    :members: multi_binary_label_cross_entropy_cost
    :noindex:

huber_cost
----------
..  automodule:: paddle.v2.layer
    :members: huber_cost
    :noindex:

lambda_cost
-----------
..  automodule:: paddle.v2.layer
    :members: lambda_cost
    :noindex:

rank_cost
---------
..  automodule:: paddle.v2.layer
    :members: rank_cost
    :noindex:

sum_cost
---------
..  automodule:: paddle.v2.layer
    :members: sum_cost
    :noindex:

crf
---
..  automodule:: paddle.v2.layer
    :members: crf
    :noindex:

crf_decoding
------------
..  automodule:: paddle.v2.layer
    :members: crf_decoding
    :noindex:

ctc
---
..  automodule:: paddle.v2.layer
    :members: ctc
    :noindex:

warp_ctc
--------
..  automodule:: paddle.v2.layer
    :members: warp_ctc
    :noindex:

nce
---
..  automodule:: paddle.v2.layer
    :members: nce
    :noindex:

hsigmoid
---------
..  automodule:: paddle.v2.layer
    :members: hsigmoid
    :noindex:

Check Layer 
============

eos
---
..  automodule:: paddle.v2.layer
    :members: eos
    :noindex:
