/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Unsqueeze.cpp - Lowering Unsqueeze Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXUnsqueezeOpLoweringToMhlo : public ConversionPattern {
  ONNXUnsqueezeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnsqueezeOpAdaptor operandAdaptor(operands);
    ONNXUnsqueezeOp unsqueezeOp = llvm::cast<ONNXUnsqueezeOp>(op);
    Location loc = op->getLoc();
    Value data = operandAdaptor.data();
    // Value axes = operandAdaptor.axes();
    // assert(isRankedShapedType(data.getType()) &&
    //        "data must be ranked Shaped Type");
    // ShapedType dataType = data.getType().cast<ShapedType>();
    // int64_t rank = dataType.getRank();

    ONNXUnsqueezeOpShapeHelper shapeHelper(&unsqueezeOp);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // SmallVector<int64_t, 4> axesList;
    // if (auto axesAttr = getDenseElementAttributeFromONNXValue(axes)) {
    //   for (IntegerAttr value : axesAttr.getValues<IntegerAttr>()) {
    //     int64_t axis = value.cast<IntegerAttr>().getInt();
    //     if (axis < 0)
    //       axis += rank;
    //     axesList.push_back(axis);
    //   }
    // }

    // int64_t newRank = rank + axesList.size();
    // SmallVector<int64_t, 4> newShape(newRank, 0);
    // for (int64_t axis : axesList) {
    //   newShape[axis] = 1;
    // }
    // for (int64_t i = 0, j = 0; i < newRank; i++) {
    //   if (newShape[i] == 0) {
    //     newShape[i] = dataType.getDimSize(j);
    //     j++;
    //   }
    // }

    Type outputType = *op->result_type_begin();
    Value newView = rewriter.create<mhlo::ReshapeOp>(loc, outputType, data);
    rewriter.replaceOp(op, newView);
    return success();
  }
};

void populateLoweringONNXUnsqueezeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
