Base
======

Layer 
-----
..  doxygenclass:: paddle::Layer
    :members:

Projection
----------
..  doxygenclass:: paddle::Projection
    :members:

Operator
--------
..  doxygenclass:: paddle::Operator
    :members:
    
Data Layer
===========

..  doxygenclass:: paddle::DataLayer
    :members:

Fully Connected Layers
======================

FullyConnectedLayer
-------------------
..  doxygenclass:: paddle::FullyConnectedLayer
    :members:

SelectiveFullyConnectedLayer
----------------------------
..  doxygenclass:: paddle::SelectiveFullyConnectedLayer
    :members:

Conv Layers
===========

ConvBaseLayer
-------------
..  doxygenclass:: paddle::ConvBaseLayer
    :members:

ConvOperator
------------
..  doxygenclass:: paddle::ConvOperator
    :members:

ConvShiftLayer
--------------
..  doxygenclass:: paddle::ConvShiftLayer
    :members:

CudnnConvLayer
--------------
..  doxygenclass:: paddle::CudnnConvLayer
    :members:

ExpandConvLayer
---------------
..  doxygenclass:: paddle::ExpandConvLayer
    :members:

ContextProjection
-----------------
..  doxygenclass:: paddle::ContextProjection
    :members:

Pooling Layers
==============

PoolLayer
---------
..  doxygenclass:: paddle::PoolLayer
    :members:

PoolProjectionLayer
-------------------
..  doxygenclass:: paddle::PoolProjectionLayer
    :members:

CudnnPoolLayer
--------------
..  doxygenclass:: paddle::CudnnPoolLayer
    :members:

Norm Layers
===========

NormLayer
---------
..  doxygenclass:: paddle::NormLayer
    :members:

CMRProjectionNormLayer
----------------------
..  doxygenclass:: paddle::CMRProjectionNormLayer
    :members:

DataNormLayer
-------------
..  doxygenclass:: paddle::DataNormLayer
    :members:

ResponseNormLayer
-----------------
..  doxygenclass:: paddle::ResponseNormLayer
    :members:

BatchNormBaseLayer
------------------
..  doxygenclass:: paddle::BatchNormBaseLayer
    :members:

BatchNormalizationLayer
-----------------------
..  doxygenclass:: paddle::BatchNormalizationLayer
    :members:

CudnnBatchNormLayer
-----------------------
..  doxygenclass:: paddle::CudnnBatchNormLayer
    :members:

SumToOneNormLayer
-----------------
..  doxygenclass:: paddle::SumToOneNormLayer
    :members:

Activation Layer
================

ParameterReluLayer
------------------
..  doxygenclass:: paddle::ParameterReluLayer
    :members:

Recurrent Layers
================

RecurrentLayer
--------------
..  doxygenclass:: paddle::RecurrentLayer
    :members:

SequenceToBatch
---------------
..  doxygenclass:: paddle::SequenceToBatch
    :members:

LSTM
----
LstmLayer
`````````
..  doxygenclass:: paddle::LstmLayer
    :members:

LstmStepLayer
`````````````
..  doxygenclass:: paddle::LstmStepLayer
    :members:

LstmCompute
```````````
..  doxygenclass:: paddle::LstmCompute
    :members:

MDLSTM
------
MDLstmLayer
```````````
..  doxygenclass:: paddle::MDLstmLayer
    :members:

CoordIterator
`````````````
..  doxygenclass:: paddle::CoordIterator
    :members:

GRU
---
GatedRecurrentLayer
```````````````````
..  doxygenclass:: paddle::GatedRecurrentLayer
    :members:

GruStepLayer
````````````
..  doxygenclass:: paddle::GruStepLayer
    :members:

GruCompute
``````````
..  doxygenclass:: paddle::GruCompute
    :members:

Recurrent Layer Group
=====================

AgentLayer
----------
..  doxygenclass:: paddle::AgentLayer
    :members:

SequenceAgentLayer
------------------
..  doxygenclass:: paddle::SequenceAgentLayer
    :members:

GatherAgentLayer
----------------
..  doxygenclass:: paddle::GatherAgentLayer
    :members:

SequenceGatherAgentLayer
------------------------
..  doxygenclass:: paddle::SequenceGatherAgentLayer
    :members:

ScatterAgentLayer
-----------------
..  doxygenclass:: paddle::ScatterAgentLayer
    :members:

SequenceScatterAgentLayer
-------------------------
..  doxygenclass:: paddle::SequenceScatterAgentLayer
    :members:

GetOutputLayer
--------------
..  doxygenclass:: paddle::GetOutputLayer
    :members:

Mixed Layer
===========
..  doxygenclass:: paddle::MixedLayer
    :members:

DotMulProjection
----------------
..  doxygenclass:: paddle::DotMulProjection
    :members:

DotMulOperator
--------------
..  doxygenclass:: paddle::DotMulOperator
    :members:

FullMatrixProjection
--------------------
..  doxygenclass:: paddle::FullMatrixProjection
    :members:

IdentityProjection
------------------
..  doxygenclass:: paddle::IdentityProjection
    :members:

IdentityOffsetProjection
------------------------
..  doxygenclass:: paddle::IdentityOffsetProjection
    :members:

TableProjection
---------------
..  doxygenclass:: paddle::TableProjection
    :members:

TransposedFullMatrixProjection
------------------------------
..  doxygenclass:: paddle::TransposedFullMatrixProjection
    :members:

Aggregate Layers
================

Aggregate
---------
AverageLayer
````````````
..  doxygenclass:: paddle::AverageLayer
    :members:

MaxLayer
````````
..  doxygenclass:: paddle::MaxLayer
    :members:

SequenceLastInstanceLayer
`````````````````````````
..  doxygenclass:: paddle::SequenceLastInstanceLayer
    :members:

Concat
------
ConcatenateLayer
````````````````
..  doxygenclass:: paddle::ConcatenateLayer
    :members:

ConcatenateLayer2
`````````````````
..  doxygenclass:: paddle::ConcatenateLayer2
    :members:

SequenceConcatLayer
```````````````````
..  doxygenclass:: paddle::SequenceConcatLayer
    :members:

Subset
------
SubSequenceLayer
````````````````
..  doxygenclass:: paddle::SubSequenceLayer
    :members:

Reshaping Layers
================

BlockExpandLayer
----------------
..  doxygenclass:: paddle::BlockExpandLayer
    :members:

ExpandLayer
-----------
..  doxygenclass:: paddle::ExpandLayer
    :members:

FeatureMapExpandLayer
---------------------
..  doxygenclass:: paddle::FeatureMapExpandLayer
    :members:

ResizeLayer
-----------
..  doxygenclass:: paddle::ResizeLayer
    :members:

SequenceReshapeLayer
--------------------
..  doxygenclass:: paddle::SequenceReshapeLayer
    :members:

Math Layers
===========

AddtoLayer
----------
..  doxygenclass:: paddle::AddtoLayer
    :members:

ConvexCombinationLayer
----------------------
..  doxygenclass:: paddle::ConvexCombinationLayer
    :members:

InterpolationLayer
------------------
..  doxygenclass:: paddle::InterpolationLayer
    :members:

MultiplexLayer
--------------
..  doxygenclass:: paddle::MultiplexLayer
    :members:

OuterProdLayer
--------------
..  doxygenclass:: paddle::OuterProdLayer
    :members:

PowerLayer
----------
..  doxygenclass:: paddle::PowerLayer
    :members:

ScalingLayer
------------
..  doxygenclass:: paddle::ScalingLayer
    :members:

SlopeInterceptLayer
-------------------
..  doxygenclass:: paddle::SlopeInterceptLayer
    :members:

TensorLayer
------------
..  doxygenclass:: paddle::TensorLayer
    :members:

TransLayer
----------
..  doxygenclass:: paddle::TransLayer
    :members:

Sampling Layers
===============

MultinomialSampler
------------------
..  doxygenclass:: paddle::MultinomialSampler
    :members:

MaxIdLayer
----------
..  doxygenclass:: paddle::MaxIdLayer
    :members:

SamplingIdLayer
---------------
..  doxygenclass:: paddle::SamplingIdLayer
    :members:

Cost Layers
===========

CostLayer
-----------
..  doxygenclass:: paddle::CostLayer
    :members:

HuberTwoClass
`````````````
..  doxygenclass:: paddle::HuberTwoClass
    :members:

LambdaCost
```````````
..  doxygenclass:: paddle::LambdaCost
    :members:

MultiBinaryLabelCrossEntropy
````````````````````````````
..  doxygenclass:: paddle::MultiBinaryLabelCrossEntropy
    :members:

MultiClassCrossEntropy
```````````````````````
..  doxygenclass:: paddle::MultiClassCrossEntropy
    :members:

MultiClassCrossEntropyWithSelfNorm
``````````````````````````````````
..  doxygenclass:: paddle::MultiClassCrossEntropyWithSelfNorm
    :members:

RankingCost
```````````
..  doxygenclass:: paddle::RankingCost
    :members:

SoftBinaryClassCrossEntropy
```````````````````````````
..  doxygenclass:: paddle::SoftBinaryClassCrossEntropy
    :members:

SumOfSquaresCostLayer
`````````````````````
..  doxygenclass:: paddle::SumOfSquaresCostLayer
    :members:

SumCostLayer
`````````````````````
..  doxygenclass:: paddle::SumCostLayer
    :members:

CosSimLayer
-----------
..  doxygenclass:: paddle::CosSimLayer
    :members:

CosSimVecMatLayer
-----------------
..  doxygenclass:: paddle::CosSimVecMatLayer
    :members:

CRFDecodingLayer
----------------
..  doxygenclass:: paddle::CRFDecodingLayer
    :members:

CRFLayer
--------
..  doxygenclass:: paddle::CRFLayer
    :members:

CTCLayer
--------
..  doxygenclass:: paddle::CTCLayer
    :members:

HierarchicalSigmoidLayer
------------------------
..  doxygenclass:: paddle::HierarchicalSigmoidLayer
    :members:

LinearChainCRF
--------------
..  doxygenclass:: paddle::LinearChainCRF
    :members:

LinearChainCTC
--------------
..  doxygenclass:: paddle::LinearChainCTC
    :members:

NCELayer
--------
..  doxygenclass:: paddle::NCELayer
    :members:

Validation Layers
-----------------

ValidationLayer
```````````````
..  doxygenclass:: paddle::ValidationLayer
    :members:

AucValidation
`````````````
..  doxygenclass:: paddle::AucValidation
    :members:

PnpairValidation
````````````````
..  doxygenclass:: paddle::PnpairValidation
    :members:

Check Layers
============

EosIdCheckLayer
---------------
..  doxygenclass:: paddle::EosIdCheckLayer
    :members:
