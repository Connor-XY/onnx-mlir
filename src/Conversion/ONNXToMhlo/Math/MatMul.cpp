/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "matmul"
static constexpr int32_t DISABLE_MAT_VEC_PRODUCT = 0;

using namespace mlir;

namespace onnx_mlir {

struct ONNXMatMulOpLoweringToMhlo : public ConversionPattern {
  ONNXMatMulOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    ONNXMatMulOp matMulOp = llvm::cast<ONNXMatMulOp>(op);
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXMatMulOp::getOperationName()),
        op->getLoc());
    ONNXMatMulOpShapeHelper shapeHelper(&matMulOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();

    Value A(operandAdaptor.A()), B(operandAdaptor.B());
    auto aRank = A.getType().cast<ShapedType>().getShape().size();
    auto bRank = B.getType().cast<ShapedType>().getShape().size();
    // Size all the arrays to padded length.
    int paddedRank = std::max(aRank, bRank);
    paddedRank = std::max(paddedRank, 2);
    DimsExpr aDims = shapeHelper.aDims;
    DimsExpr bDims = shapeHelper.bDims;
    llvm::BitVector aPadDims = shapeHelper.aPadDims;
    llvm::BitVector bPadDims = shapeHelper.bPadDims;

    DimsExpr outputDims = shapeHelper.dimsForOutput();
    llvm::SmallVector<int64_t, 4> aShape;
    llvm::SmallVector<int64_t, 4> bShape;
    for (int64_t i = 0; i < paddedRank - 2; i++) {
      aShape.push_back(outputDims[i].isLiteral() ? outputDims[i].getLiteral()
                                                 : ShapedType::kDynamicSize);
      bShape.push_back(outputDims[i].isLiteral() ? outputDims[i].getLiteral()
                                                 : ShapedType::kDynamicSize);
    }
    if (!aPadDims[paddedRank - 2])
      aShape.push_back(aDims[paddedRank - 2].isLiteral()
                           ? aDims[paddedRank - 2].getLiteral()
                           : ShapedType::kDynamicSize);
    aShape.push_back(aDims[paddedRank - 1].isLiteral()
                         ? aDims[paddedRank - 1].getLiteral()
                         : ShapedType::kDynamicSize);
    bShape.push_back(bDims[paddedRank - 2].isLiteral()
                         ? bDims[paddedRank - 2].getLiteral()
                         : ShapedType::kDynamicSize);
    if (!bPadDims[paddedRank - 1])
      bShape.push_back(bDims[paddedRank - 1].isLiteral()
                           ? bDims[paddedRank - 1].getLiteral()
                           : ShapedType::kDynamicSize);
    Type outputAType = RankedTensorType::get(aShape, elementType);
    Type outputBType = RankedTensorType::get(bShape, elementType);

    int64_t oneDPadA = aPadDims[paddedRank - 2];
    int64_t oneDPadB = bPadDims[paddedRank - 1];

    Value broadcastedA;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadA - aRank, paddedRank - oneDPadA));
      broadcastedA = rewriter.createOrFold<mhlo::BroadcastInDimOp>(
          loc, outputAType, A, rewriter.getI64VectorAttr(broadcastDimensions));
    }
    Value broadcastedB;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadB - bRank, paddedRank - oneDPadB));
      broadcastedB = rewriter.createOrFold<mhlo::BroadcastInDimOp>(
          loc, outputBType, B, rewriter.getI64VectorAttr(broadcastDimensions));
    }
    Value dotProduct;
    if (paddedRank > 2) {
      dotProduct = rewriter.create<mhlo::DotGeneralOp>(loc, outputType,
          broadcastedA, broadcastedB,
          mhlo::DotDimensionNumbersAttr::get(rewriter.getContext(),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              {paddedRank - 1 - oneDPadA}, {paddedRank - 2}),
          nullptr);
    } else {
      dotProduct = rewriter.create<mhlo::DotOp>(
          loc, broadcastedA, broadcastedB, nullptr);
      if (aRank == 1 && bRank == 1)
        dotProduct = rewriter.create<mhlo::BroadcastInDimOp>(loc, outputType,
            dotProduct, rewriter.getI64TensorAttr({}));
    }
    rewriter.replaceOp(op, dotProduct);
    return success();
  }
};

void populateLoweringONNXMatMulOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
