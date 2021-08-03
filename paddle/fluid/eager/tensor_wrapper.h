/**
#pragma once

#include "paddle/pten/core/tensor.h"

namespace egr {

 * Tensor Wrapper is designed to be a holder for Tensor, in eager mode we need to let
 * GradOpNodes to hold forward tensor as its input. However, those forward Tensor is 
 * designed to hold AutogradMeta which will hold its GradOpNodes. Apparently, this will
 * cause reference cycles. problem and make those forward tensors can't be released
 * in time. 
 * 
 * Therefore, we need this wrapper to hold forward tensor's data instead of auto grad info.
 * In this way, we can break the reference cycles we talked before.
 * 
 * TODO:(jiabin) Should we copy AutogradMeta info into TensorWrapper or not?
 * 
 *  **/

/**
 * It looks like we don't need Tensor wrapper now with current Tensor structure, we can
 * just assigned one tensor to another which only shared tensor_impl
 *  
 *  class TensorWrapper{
 *  public:
 *  explicit TensorWrapper(const Tensor& tensor){
 *      
 *  }

 *  private:

 *  Tensor data_;

 *  }
} // namespace egr

**/