/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The gradient of a whole ONNX model, built from graph_grad.h's rules. See
 * graph_grad_model.cpp for the steps; the C entry point is
 * onnxsim_gradient_model in capi/onnxsim_c_api.h.
 */
#pragma once

#include <onnx/onnx_pb.h>

#include <string>
#include <vector>

// Returns the forward nodes `loss` needs, followed by the nodes computing
// d loss / d wrt[k] (graph_grad.h's rules), as a new model.
//
// `loss` names a scalar float32 or float64 value of `forward`. Every `wrt[k]`
// is a value of `forward` (an input, an initializer or a node output) with the
// loss's element type. Each of them, and every tensor the differentiated nodes
// touch, must have a static shape: the shapes come from onnx shape inference.
//
// The model keeps `forward`'s inputs, initializers, opset imports and
// model-local functions. It has one graph output `<prefix>grad_<k>` per wrt,
// in order, with the element type and shape of wrt[k]. Every new tensor name
// starts with `prefix`, so the model can be merged into a graph that already
// holds `forward` by name. Two calls with different prefixes give disjoint
// new names; two calls with the same prefix give identical ones, so choose
// distinct prefixes for models that will live in one graph.
//
// The gradient is built in the element type of the loss. A wrt the loss does
// not reach gets a zero of its shape; a wrt that is the loss gets the seed 1.
// Only nodes that lie between a wrt and the loss are differentiated, so an
// operator without a rule that the loss does not depend on is not an error.
//
// Throws UnsupportedOpError (graph_grad.h) for a differentiated node without a
// rule, and std::invalid_argument for a bad loss, wrt or missing shape.
onnx::ModelProto BuildGradientModel(const onnx::ModelProto& forward,
                                    const std::string& loss,
                                    const std::vector<std::string>& wrt,
                                    const std::string& prefix = "grad_");
