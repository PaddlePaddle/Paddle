..  _api_v2.layer:

======
Layers
======

Data layer
===========

..  _api_v2.layer_data:

data
----
..  autofunction:: paddle.v2.layer.data
    :noindex:

Fully Connected Layers
======================

..  _api_v2.layer_fc:

fc
--
..  autofunction:: paddle.v2.layer.fc
    :noindex:

selective_fc
------------
..  autofunction:: paddle.v2.layer.selective_fc
    :noindex:

Conv Layers
===========

conv_operator
-------------
..  autofunction:: paddle.v2.layer.conv_operator
    :noindex:

conv_projection
---------------
..  autofunction:: paddle.v2.layer.conv_projection
    :noindex:

conv_shift
----------
..  autofunction:: paddle.v2.layer.conv_shift
    :noindex:

img_conv
--------
..  autofunction:: paddle.v2.layer.img_conv
    :noindex:

..  _api_v2.layer_context_projection:

context_projection
------------------
..  autofunction:: paddle.v2.layer.context_projection
    :noindex:

row_conv
--------
..  autofunction:: paddle.v2.layer.row_conv
    :noindex:

Image Pooling Layer
===================

img_pool
--------
..  autofunction:: paddle.v2.layer.img_pool
    :noindex:

spp
---
..  autofunction:: paddle.v2.layer.spp
    :noindex:

maxout
------
..  autofunction:: paddle.v2.layer.maxout
    :noindex:

roi_pool
--------
..  autofunction:: paddle.v2.layer.roi_pool
    :noindex:

pad
----
..  autofunction:: paddle.v2.layer.pad
    :noindex:

Norm Layer
==========

img_cmrnorm
-----------
..  autofunction:: paddle.v2.layer.img_cmrnorm
    :noindex:

batch_norm
----------
..  autofunction:: paddle.v2.layer.batch_norm
    :noindex:

sum_to_one_norm
---------------
..  autofunction:: paddle.v2.layer.sum_to_one_norm
    :noindex:

cross_channel_norm
------------------
..  autofunction:: paddle.v2.layer.cross_channel_norm
    :noindex:

row_l2_norm
-----------
..  autofunction:: paddle.v2.layer.row_l2_norm
    :noindex:

Recurrent Layers
================

recurrent
---------
..  autofunction:: paddle.v2.layer.recurrent
    :noindex:

lstmemory
---------
..  autofunction:: paddle.v2.layer.lstmemory
    :noindex:

grumemory
---------
..  autofunction:: paddle.v2.layer.grumemory
    :noindex:

gated_unit
-----------
..  autofunction:: paddle.v2.layer.gated_unit
    :noindex:

Recurrent Layer Group
=====================

memory
------
..  autofunction:: paddle.v2.layer.memory
    :noindex:

recurrent_group
---------------
..  autofunction:: paddle.v2.layer.recurrent_group
    :noindex:

lstm_step
---------
..  autofunction:: paddle.v2.layer.lstm_step
    :noindex:

gru_step
--------
..  autofunction:: paddle.v2.layer.gru_step
    :noindex:

beam_search
------------
..  autofunction:: paddle.v2.layer.beam_search
    :noindex:

get_output
----------
..  autofunction:: paddle.v2.layer.get_output
    :noindex:

Mixed Layer
===========

..  _api_v2.layer_mixed:

mixed
-----
..  autofunction:: paddle.v2.layer.mixed
    :noindex:

..  _api_v2.layer_embedding:

embedding
---------
..  autofunction:: paddle.v2.layer.embedding
    :noindex:

scaling_projection
------------------
..  autofunction:: paddle.v2.layer.scaling_projection
    :noindex:

dotmul_projection
-----------------
..  autofunction:: paddle.v2.layer.dotmul_projection
    :noindex:

dotmul_operator
---------------
..  autofunction:: paddle.v2.layer.dotmul_operator
    :noindex:

full_matrix_projection
----------------------
..  autofunction:: paddle.v2.layer.full_matrix_projection
    :noindex:

identity_projection
-------------------
..  autofunction:: paddle.v2.layer.identity_projection
    :noindex:

slice_projection
-------------------
..  autofunction:: paddle.v2.layer.slice_projection
    :noindex:

table_projection
----------------
..  autofunction:: paddle.v2.layer.table_projection
    :noindex:

trans_full_matrix_projection
----------------------------
..  autofunction:: paddle.v2.layer.trans_full_matrix_projection
    :noindex:

Aggregate Layers
================

AggregateLevel
--------------
..  autoclass:: paddle.v2.layer.AggregateLevel
    :noindex:

..  _api_v2.layer_pooling:

pooling
-------
..  autofunction:: paddle.v2.layer.pooling
    :noindex:

..  _api_v2.layer_last_seq:

last_seq
--------
..  autofunction:: paddle.v2.layer.last_seq
    :noindex:

..  _api_v2.layer_first_seq:

first_seq
---------
..  autofunction:: paddle.v2.layer.first_seq
    :noindex:

sub_seq
---------
..  autofunction:: paddle.v2.layer.sub_seq
    :noindex:

concat
------
..  autofunction:: paddle.v2.layer.concat
    :noindex:

seq_concat
----------
..  autofunction:: paddle.v2.layer.seq_concat
    :noindex:

seq_slice
---------
..  autofunction:: paddle.v2.layer.seq_slice
    :noindex:

sub_nested_seq
--------------
..  autofunction:: paddle.v2.layer.sub_nested_seq
    :noindex:

Reshaping Layers
================

block_expand
------------
..  autofunction:: paddle.v2.layer.block_expand
    :noindex:

..  _api_v2.layer_expand:

ExpandLevel
-----------
..  autoclass:: paddle.v2.layer.ExpandLevel
    :noindex:

expand
------
..  autofunction:: paddle.v2.layer.expand
    :noindex:

repeat
------
..  autofunction:: paddle.v2.layer.repeat
    :noindex:

rotate
------
..  autofunction:: paddle.v2.layer.rotate
    :noindex:

seq_reshape
-----------
..  autofunction:: paddle.v2.layer.seq_reshape
    :noindex:

Math Layers
===========

addto
-----
..  autofunction:: paddle.v2.layer.addto
    :noindex:

linear_comb
-----------
..  autofunction:: paddle.v2.layer.linear_comb
    :noindex:

interpolation
-------------
..  autofunction:: paddle.v2.layer.interpolation
    :noindex:

bilinear_interp
---------------
..  autofunction:: paddle.v2.layer.bilinear_interp
    :noindex:

dropout
--------
..  autofunction:: paddle.v2.layer.dropout
    :noindex:

dot_prod
---------
.. autofunction:: paddle.v2.layer.dot_prod
    :noindex:

out_prod
--------
.. autofunction:: paddle.v2.layer.out_prod
    :noindex:

power
-----
..  autofunction:: paddle.v2.layer.power
    :noindex:

scaling
-------
..  autofunction:: paddle.v2.layer.scaling
    :noindex:

clip
----
..  autofunction:: paddle.v2.layer.clip
    :noindex:

resize
------
..  autofunction:: paddle.v2.layer.resize
    :noindex:

slope_intercept
---------------
..  autofunction:: paddle.v2.layer.slope_intercept
    :noindex:

tensor
------
..  autofunction:: paddle.v2.layer.tensor
    :noindex:

..  _api_v2.layer_cos_sim:

cos_sim
-------
..  autofunction:: paddle.v2.layer.cos_sim
    :noindex:

l2_distance
-----------
..  autofunction:: paddle.v2.layer.l2_distance
    :noindex:

trans
-----
..  autofunction:: paddle.v2.layer.trans
    :noindex:

scale_shift
-----------
..  autofunction:: paddle.v2.layer.scale_shift
    :noindex:

factorization_machine
---------------------
..  autofunction:: paddle.v2.layer.factorization_machine
    :noindex:

Sampling Layers
===============

maxid
-----
..  autofunction:: paddle.v2.layer.max_id
    :noindex:

sampling_id
-----------
..  autofunction:: paddle.v2.layer.sampling_id
    :noindex:

multiplex
---------
..  autofunction:: paddle.v2.layer.multiplex
    :noindex:

..  _api_v2.layer_costs:

Cost Layers
===========

cross_entropy_cost
------------------
..  autofunction:: paddle.v2.layer.cross_entropy_cost
    :noindex:

cross_entropy_with_selfnorm_cost
--------------------------------
..  autofunction:: paddle.v2.layer.cross_entropy_with_selfnorm_cost
    :noindex:

multi_binary_label_cross_entropy_cost
-------------------------------------
..  autofunction:: paddle.v2.layer.multi_binary_label_cross_entropy_cost
    :noindex:

classification_cost
-------------------
.. autofunction:: paddle.v2.layer.classification_cost
   :noindex:

huber_regression_cost
-------------------------
..  autofunction:: paddle.v2.layer.huber_regression_cost
    :noindex:

huber_classification_cost
-------------------------
..  autofunction:: paddle.v2.layer.huber_classification_cost
    :noindex:

lambda_cost
-----------
..  autofunction:: paddle.v2.layer.lambda_cost
    :noindex:

square_error_cost
-----------------
..  autofunction:: paddle.v2.layer.square_error_cost
    :noindex:

rank_cost
---------
..  autofunction:: paddle.v2.layer.rank_cost
    :noindex:

sum_cost
---------
..  autofunction:: paddle.v2.layer.sum_cost
    :noindex:

crf
---
..  autofunction:: paddle.v2.layer.crf
    :noindex:

crf_decoding
------------
..  autofunction:: paddle.v2.layer.crf_decoding
    :noindex:

ctc
---
..  autofunction:: paddle.v2.layer.ctc
    :noindex:

warp_ctc
--------
..  autofunction:: paddle.v2.layer.warp_ctc
    :noindex:

nce
---
..  autofunction:: paddle.v2.layer.nce
    :noindex:

hsigmoid
---------
..  autofunction:: paddle.v2.layer.hsigmoid
    :noindex:

smooth_l1_cost
--------------
..  autofunction:: paddle.v2.layer.smooth_l1_cost
    :noindex:

multibox_loss
--------------
..  autofunction:: paddle.v2.layer.multibox_loss
    :noindex:

detection_output
----------------
..  autofunction:: paddle.v2.layer.detection_output
    :noindex:

Check Layer
============

eos
---
..  autofunction:: paddle.v2.layer.eos
    :noindex:

Activation
==========

prelu
--------
..  autofunction:: paddle.v2.layer.prelu
    :noindex:
