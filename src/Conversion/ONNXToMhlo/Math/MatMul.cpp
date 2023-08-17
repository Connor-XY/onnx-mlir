/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

int64_t getLiteralValue(const IndexExpr &idx) {
  return idx.isLiteral() ? idx.getLiteral() : ShapedType::kDynamic;
}

struct ONNXMatMulOpLoweringToMhlo : public ConversionPattern {
  ONNXMatMulOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    IndexExprBuilderForMhlo createIE(rewriter, loc);
    ONNXMatMulOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();

    Value A(operandAdaptor.getA()), B(operandAdaptor.getB());
    auto aRank = A.getType().cast<ShapedType>().getShape().size();
    auto bRank = B.getType().cast<ShapedType>().getShape().size();
    // Size all the arrays to padded length.
    int paddedRank = std::max(aRank, bRank);
    paddedRank = std::max(paddedRank, 2);
    DimsExpr aDims = shapeHelper.aDims;
    DimsExpr bDims = shapeHelper.bDims;
    llvm::BitVector aPadDims = shapeHelper.aPadDims;
    llvm::BitVector bPadDims = shapeHelper.bPadDims;

    DimsExpr outputDims = shapeHelper.getOutputDims();
    llvm::SmallVector<int64_t, 4> aShapeList;
    llvm::SmallVector<int64_t, 4> bShapeList;
    llvm::SmallVector<int64_t, 4> outputShapeList;

    IndexExpr::getShape(outputDims, outputShapeList);
    IndexExpr::getShape(aDims, aShapeList);
    IndexExpr::getShape(bDims, bShapeList);

    llvm::SmallVector<int64_t, 4> aShape;
    llvm::SmallVector<int64_t, 4> bShape;
    Value AShape = rewriter.create<shape::ShapeOfOp>(loc, A);
    Value BShape = rewriter.create<shape::ShapeOfOp>(loc, B);
    SmallVector<Value> AShapeDims;
    SmallVector<Value> BShapeDims;
    int aOffset = paddedRank - aRank;
    int bOffset = paddedRank - bRank;
    if (bRank == 1)
      bOffset--;
    for (int64_t i = 0; i < paddedRank - 2; i++) {
      aShape.push_back(outputShapeList[i]);
      bShape.push_back(outputShapeList[i]);
      if (outputShapeList[i] != ShapedType::kDynamic) { // use output dim
        AShapeDims.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, outputShapeList[i]));
        BShapeDims.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, outputShapeList[i]));
      } else if (aDims[i].isLiteralAndIdenticalTo(1)) { // use B dim
        AShapeDims.push_back(
            rewriter.create<shape::GetExtentOp>(loc, BShape, i));
        BShapeDims.push_back(
            rewriter.create<shape::GetExtentOp>(loc, BShape, i));
      } else { // use A dim
        AShapeDims.push_back(
            rewriter.create<shape::GetExtentOp>(loc, AShape, i));
        BShapeDims.push_back(
            rewriter.create<shape::GetExtentOp>(loc, AShape, i));
      }
    }
    if (!aPadDims[paddedRank - 2]) {
      aShape.push_back(aShapeList[paddedRank - 2]);
      AShapeDims.push_back(rewriter.create<shape::GetExtentOp>(
          loc, AShape, paddedRank - 2 - aOffset));
    }
    aShape.push_back(aShapeList[paddedRank - 1]);
    bShape.push_back(bShapeList[paddedRank - 2]);
    AShapeDims.push_back(rewriter.create<shape::GetExtentOp>(
        loc, AShape, paddedRank - 1 - aOffset));
    BShapeDims.push_back(rewriter.create<shape::GetExtentOp>(
        loc, BShape, paddedRank - 2 - bOffset));
    if (!bPadDims[paddedRank - 1]) {
      bShape.push_back(bShapeList[paddedRank - 1]);
      BShapeDims.push_back(rewriter.create<shape::GetExtentOp>(
          loc, BShape, paddedRank - 1 - bOffset));
    }

    Type outputAType = RankedTensorType::get(aShape, elementType);
    Type outputBType = RankedTensorType::get(bShape, elementType);
    Type outputAShapeType =
        RankedTensorType::get({aShape.size()}, rewriter.getIndexType());
    Type outputBShapeType =
        RankedTensorType::get({bShape.size()}, rewriter.getIndexType());
    Value outputAShape = rewriter.create<shape::FromExtentsOp>(loc, AShapeDims);
    outputAShape = rewriter.create<shape::ToExtentTensorOp>(
        loc, outputAShapeType, outputAShape);
    Value outputBShape = rewriter.create<shape::FromExtentsOp>(loc, BShapeDims);
    outputBShape = rewriter.create<shape::ToExtentTensorOp>(
        loc, outputBShapeType, outputBShape);

    int64_t oneDPadA = aPadDims[paddedRank - 2];
    int64_t oneDPadB = bPadDims[paddedRank - 1];

    Value broadcastedA;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadA - aRank, paddedRank - oneDPadA));
      broadcastedA =
          rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(loc, outputAType,
              A, outputAShape, rewriter.getI64VectorAttr(broadcastDimensions));
    }
    Value broadcastedB;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadB - bRank, paddedRank - oneDPadB));
      broadcastedB =
          rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(loc, outputBType,
              B, outputBShape, rewriter.getI64VectorAttr(broadcastDimensions));
    }
    Value dotProduct;
    if (paddedRank > 2)
      dotProduct = rewriter.create<mhlo::DotGeneralOp>(loc, outputType,
          broadcastedA, broadcastedB,
          mhlo::DotDimensionNumbersAttr::get(rewriter.getContext(),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              {paddedRank - 1 - oneDPadA}, {paddedRank - 2}),
          nullptr);
    else {
      dotProduct = rewriter.create<mhlo::DotOp>(loc,
          op->getResultTypes().front(), broadcastedA, broadcastedB, nullptr);
    }
    rewriter.replaceOp(op, dotProduct);
    return success();
  }
};

} // namespace

void populateLoweringONNXMatMulOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
