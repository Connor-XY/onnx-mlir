/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXFlattenOpLoweringToMhlo : public ConversionPattern {
  ONNXFlattenOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXFlattenOpAdaptor operandAdaptor(operands);
    ONNXFlattenOp flattenOp = llvm::cast<ONNXFlattenOp>(op);

    Value input = operandAdaptor.input();
    auto inputType = input.getType().cast<RankedTensorType>();
    if (inputType == nullptr) {
      op->emitError() << "Flatten Output Is Not Ranked\n";
      return failure();
    }
    auto rank = inputType.getRank();
    int64_t axis = flattenOp.axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);
    
    Value flattenDimFirst = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (int64_t i = 0; i < axis; i++) {
      Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
      flattenDimFirst = rewriter.create<arith::MulIOp>(loc, flattenDimFirst, dim);
    }
    Value flattenDimSecond = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (int64_t i = axis; i < rank; i++) {
      Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
      flattenDimSecond = rewriter.create<arith::MulIOp>(loc, flattenDimSecond, dim);
    }
    SmallVector<Value> dims{flattenDimFirst, flattenDimSecond};
    Type elementType =
        RankedTensorType::get({2}, rewriter.getIndexType());
    Value outputShape =
        rewriter.create<tensor::FromElementsOp>(loc, elementType, dims);
    auto result = rewriter.create<mhlo::DynamicReshapeOp>(
      loc, *op->result_type_begin(), input, outputShape);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
