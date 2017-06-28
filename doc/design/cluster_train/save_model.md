# Design Doc: Save Model

## Overview

The model is the output of the training process. There are two
ways from which user can obtain a model:

- Save model triggered by user code: user code asks PaddlePaddle to
  save a model.
- Convert model from the snapshot: model being converted from
  pservers' periodic snapshot. In this way, the user can cancel a job
  at any time, and still have a relatively fresh model (we snapshot
  around every 5 minutes).

### Save Model Triggered by User Code

Both trainers and pservers have access to the model. So the model can
be saved from a trainer or pservers. We need to decide on where the
model is saved from.

#### Dense Model vs. Sparse Model

There are two types of model: dense and sparse model (when the
parameter is configured to be sparse). Pservers always jointly have
the entire model at any given time. Trainers only have the entire
dense model, but only have a fraction of the sparse model at any given
time.

#### Pservers Saving Model

The benefit of letting pservers save model is they have the entire
model all the time. However, since pservers are on different nodes, it
requires a merging process to merge model shards into the same
model. Thus requires the pservers to write models to a distributed
filesystem, making the snapshot shards visible to the merge program.

#### Trainer Saving Model

The benefit of letting one trainer to save the model is it does not
require a distributed filesystem. And it's reusing the same save model
logic when the trainer is training locally - except when training
sparse model, the trainer needs to download the entire sparse model
during the saving process.

#### Conclusion

Given trainer saving model does not require a distributed filesystem,
and is an intuitive extension to training locally, we decide to let
the trainer save the model.


### Convert Model from Snapshot

TODO


## Timeline

We first implement trainer save the model. Converting the latest
snapshot to a model will be a TODO for future.


## Trainer Save Model

### Trainer Election

One trainer will be elected as the one to save the model. When using
etcd, trainer ID is a randomly generated UUID, we will utilize etcd to
elect one trainer. When not using etcd, unique trainer IDs will be
given by the administrator, the trainer whose ID is "0" is elected to
save the model.

### Model Save Path

Each trainer will be given the directory to save the model. The
elected trainer will save the model to
`given-directory/trainerID`. Since the tainerID is unique, this would
prevent concurrent save to the same file when multiple trainers are
elected to save the model when split-brain problem happens.

### What Happens When Model Is Saving

It takes some time to save model, we need to define what will happen
when save model is taking place.

When saving a dense model, the trainer uses the local model. Pservers
does not need to pause model update.

When saving a sparse model. The trainer needs to download the entire
sparse model while saving. To get the most accurate model, the model
update needs to be paused before the download starts and resumed after
the download finishes. Otherwise, the trainer gets a model that is
"polluted": some part of the model is old, some part of the model is
new.

It's unclear that the "polluted" model will be inferiod due to the
stochastic nature of deep learning, and pausing the model update will
add more complexity to the system. Since supporting sparse model is a
TODO item. We defer the evaluation of pause the model update or not
during saving model to the future.
