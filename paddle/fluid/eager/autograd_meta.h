#pragma once

#include "paddle/pten/core/tensor.h"

namespace egr {

/**
 * 
 * AutogradMeta is what record the backward info for tensor. When we run computation
 * graph eagerly, we can not build a static paddle program like static mode do, so we
 * need a new method to record forward info to trace backward when we finish all forward
 * computation. This require our AutogradMeta class record following main members
 * 
 * 1. grad_op: 
 * Grad_op indicate the grad operation of the forward op
 * 
 * 2. grad:
 * Grad is the gradient of forward Tensor, which should be compute after backward computation
 * 
 * NOTE: grad should only be available when current tensor is a leaf tensor, and for non-leaf
 * tensor grad is only available while user set `retain_grad` option as `true`.
 * 
 * TODO:(jiabin) support hooks 
 * 3. hooks:
 * Hooks are some computation logic which only attached with backward operation, it registered
 * by user and run before accumulator.
 * 
 * 4.overrided_stop_gradient_
 * This member is used to finish some auto-prune related work, which indicate user set stop_gradient
 * should overrided the result indicated by framework. All non-parameter tensor's stop_gradient
 * properties should be true. We will pass stop_gradient when we find one who need it.
 * 
 * NOTE: AutogradMeta is inherited from AutogradMetaInterface which is defined in tensor's deps,
 * we did this to avoid additional dependency on Autograd. In eager execution, we will cast
 * AutogradMetaInterface as AutogradMeta to use it.
 * 
 * **/

class AutogradMeta : public AutogradMetaInterface{
public:
    AutogradMeta(const Edge& edge = Edge()){
        output_rank_ = -1;
        grad_node_ = edge.GetGradNode();
    }

    ~AutogradMeta() override = default;

    const Tensor& Grad() const {
        return grad_;
    }
    void SetGradNode(std::shared_ptr<GradNodeBase> grad_node){
        grad_node_ = grad_node;
    }
    std::share_ptr<GradNodeBase> GradNode() const {
        return grad_node_;
    };
    
    void SetOutRank(size_t rank){
        output_rank_ = rank;
    }

    size_t OutRank() const { return output_rank_; }

    bool IsInitialized() { return !grad_node_.get(); }

private:
    // TODO(jiabin): Should we use pointer instead of object? 
    pt::Tensor grad_;
    
    // GradNodeBase is base class of all grad op which is a 
    // wrapper for grad op. This class will make grad op easy 
    // to be traced.
    std::shared_ptr<GradNodeBase> grad_node_;
    
    // output rank of forward op, this is a vital num, since 
    // we are now trying to make our forward output is as same
    // sequence as backward input. In case of tracing backward
    // sequence we need to record output rank here.
    size_t output_rank_;
    
    // TODO:(jiabin) Support hooks here and store it in AutogradMeta
    
    // Stop gradient flag to indicate should we compute backward
    int overrided_stop_gradient_{-1};
    
    bool persistable_{false};

    // TODO:(jiabin) Support Quantum here and add cache mechanism as 
    // VarCache defined in VarBase
}
    
} // namespace egr