/*
 * This file contains all the flags that declared in Node::Attr.
 *
 * The Node::Attr is designed to share information between different passes, one
 * can get other's attributes in a Node by the flags in this file.
 */
#pragma once
namespace paddle {
namespace inference {
namespace analysis {

#define DECLARE_NODE_ATTR(flag__) const char ATTR_##flag__[] = #flag__;

DECLARE_NODE_ATTR(supported_by_tensorrt)  // bool

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
