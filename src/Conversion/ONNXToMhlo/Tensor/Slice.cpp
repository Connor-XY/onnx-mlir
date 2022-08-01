/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Slice.cpp - Lowering Slice Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSliceOpLoweringToMhlo : public ConversionPattern {
  ONNXSliceOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    Location loc = op->getLoc();

    ONNXSliceOpShapeHelper shapeHelper(&sliceOp);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();
    int64_t outputRank = outputShapedType.getShape().size();

    
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSliceOpToMhloPattern(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
