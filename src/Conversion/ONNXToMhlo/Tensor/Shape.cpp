/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Shape.cpp - Lowering Shape Op ----------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXShapeOpLoweringToMhlo : public ConversionPattern {
  ONNXShapeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXShapeOpAdaptor operandAdaptor(operands);
    ONNXShapeOp shapeOp = cast<ONNXShapeOp>(op);
    Location loc = op->getLoc();
    ONNXShapeOpShapeHelper shapeHelper(&shapeOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();
    Type resultOutputType = RankedTensorType::get(
        shapeHelper.dimsForOutput(0)[0].getLiteral(), elementType);

    // Compute the data selected by the Shape operator.
    // DimsExpr selectedData = computeSelectedData(operandAdaptor);
    // llvm::SmallVector<Value, 4> selectedValue;
    // for (uint64_t i = 0; i < selectedData.size(); ++i) {
    //   Value val = selectedData[i].getValue();
    //   selectedValue.push_back(val);
    // }
    // Value concat = rewriter.create<mhlo::ConcatenateOp>(
    //     loc, resultOutputType, selectedValue, rewriter.getI64IntegerAttr({}));
    // rewriter.replaceOp(op, concat);
    Value input = shapeOp.data();
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, input);
    Value castedShape = rewriter.create<arith::IndexCastOp>(loc, resultOutputType, shape);
    rewriter.replaceOp(op, castedShape);
    // llvm::outs() << *op->getParentOp() << "\n";
    return success();
  }
};

void populateLoweringONNXShapeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
