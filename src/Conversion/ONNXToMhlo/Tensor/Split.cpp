/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSplitOp(A) is implemented using MHLO sliceOp
struct ONNXSplitOpLoweringToMhlo : public ConversionPattern {
  ONNXSplitOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSplitOpAdaptor operandAdaptor(operands);
    ONNXSplitOp splitOp = llvm::cast<ONNXSplitOp>(op);
    Value input = splitOp.getInput();
    Value split = splitOp.getSplit();
    assert(isRankedShapedType(input.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type indiceType = rewriter.getI64Type();
    Location loc = op->getLoc();
    uint64_t rank = inputType.getRank();
    uint64_t outputNum = splitOp.getNumResults();
    int64_t dimIndex = splitOp.getAxis();
    if (dimIndex < 0)
      dimIndex += rank;
    int64_t inputDimSize = inputType.getDimSize(dimIndex);

    // Get a shape helper (not used?)
    IndexExprBuilderForMhlo createIE(rewriter, loc);
    ONNXSplitOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t, 4> splitSizes;
    if (auto splitAttr = getElementAttributeFromONNXValue(split)) {
      for (IntegerAttr value : splitAttr.getValues<IntegerAttr>()) {
        int64_t splitSize = value.cast<IntegerAttr>().getInt();
        splitSizes.push_back(splitSize);
      }
    } else if (split.getType().template isa<NoneType>()) {
      assert(!ShapedType::isDynamic(inputDimSize) &&
             "input dim size can't be dynamic");
      int64_t floorSliceSize = inputDimSize / outputNum;
      unsigned numBiggerChunks = inputDimSize % outputNum;
      int64_t ceilSliceSize = floorSliceSize + (numBiggerChunks ? 1 : 0);
      for (unsigned int i = 0; i < numOfResults; ++i) {
        if (i < numBiggerChunks)
          splitSizes.push_back(ceilSliceSize);
        else
          splitSizes.push_back(floorSliceSize);
      }
    } else {
      assert(false && "dynamic split not yet supported");
    }

    SmallVector<Value, 4> slices;
    slices.reserve(outputNum);
    int64_t beginIndice = 0;
    int64_t endIndice = 0;

    if (inputType.hasStaticShape()) {
      SmallVector<int64_t, 4> sliceShape =
        llvm::to_vector<4>(inputType.getShape());
      SmallVector<int64_t, 4> beginIndices(rank, 0);
      SmallVector<int64_t, 4> endIndices =
          llvm::to_vector<4>(inputType.getShape());
      SmallVector<int64_t, 4> strides(rank, 1);

      for (uint64_t i = 0; i < outputNum; ++i) {
        sliceShape[dimIndex] = splitSizes[i];
        Type sliceType =
            RankedTensorType::get(sliceShape, inputType.getElementType());
        endIndice += splitSizes[i];
        beginIndices[dimIndex] = beginIndice;
        endIndices[dimIndex] = endIndice;
        slices.push_back(rewriter.create<mhlo::SliceOp>(loc, sliceType, input,
            DenseIntElementsAttr::get(
                RankedTensorType::get(
                    {static_cast<int64_t>(beginIndices.size())}, indiceType),
                beginIndices),
            DenseIntElementsAttr::get(
                RankedTensorType::get(
                    {static_cast<int64_t>(endIndices.size())}, indiceType),
                endIndices),
            DenseIntElementsAttr::get(
                RankedTensorType::get(
                    {static_cast<int64_t>(strides.size())}, indiceType),
                strides)));
        beginIndice = endIndice;
      }
      rewriter.replaceOp(op, slices);
      return success();
    } 
    
    Value zero = rewriter.create<mhlo::ConstantOp>(loc,
        DenseIntElementsAttr::get(RankedTensorType::get({1}, indiceType),
            ArrayRef<int64_t>{0}));
    Value one = rewriter.create<mhlo::ConstantOp>(loc,
        DenseIntElementsAttr::get(RankedTensorType::get({1}, indiceType),
            ArrayRef<int64_t>{1}));
    SmallVector<Value, 4> stepValues;
    SmallVector<Value, 4> beginValues;
    SmallVector<Value, 4> endValues;
    for (int64_t i = 0; i < rank; ++i) {
      beginValues.push_back(zero);
      stepValues.push_back(one);
      if ()
      endValues.push_back();
    }
    for (uint64_t i = 0; i < outputNum; ++i) {
      endIndice += splitSizes[i];
      slices.push_back(rewriter.create<mhlo::RealDynamicSliceOp>(loc, outputType, input,
          start_indices, end_indices, step_indices));
    }
    rewriter.replaceOp(op, slices);
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
