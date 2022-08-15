/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Conv Operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConvOpLoweringToMhlo : public ConversionPattern {
  ONNXConvOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);
    Location loc = op->getLoc();

    ONNXConvOpShapeHelper shapeHelper(&convOp);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    auto kernelShape = shapeHelper.kernelShape;
    auto strides = shapeHelper.strides;
    auto dilations = shapeHelper.dilations;
    auto outputDims = shapeHelper.dimsForOutput();
    int outputRank = shapeHelper.dimsForOutput().size();

    Value inputOperand = operandAdaptor.X();
    Value filterOperand = operandAdaptor.W();
    Value biasOperand = operandAdaptor.B();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.group();

    RankedTensorType inputType =
        inputOperand.getType().dyn_cast_or_null<RankedTensorType>();
    if (inputType == nullptr) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    Type outputType = *op->result_type_begin();
    // Onnx Input is NCHW
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t kernelSize = kernelShape.size();

    SmallVector<int64_t> inputSpatialDimensions;
    for (int64_t i = spatialOffset; i < rank; i++) {
      inputSpatialDimensions.push_back(i);
    }

    SmallVector<int64_t> kernelDimensions;
    for (int64_t i = spatialOffset; i < spatialOffset + kernelSize; i++) {
      kernelDimensions.push_back(i);
    }

    SmallVector<int64_t> outputSpatialDimensions;
    for (int64_t i = spatialOffset; i < outputRank; i++) {
      outputSpatialDimensions.push_back(i);
    }

    // paddings
    auto pads = shapeHelper.pads;
    auto padding = convOp.auto_pad();
    int64_t spatialRank = rank - spatialOffset;
    SmallVector<int64_t> flattenPaddings;
    bool needPadding = (padding == "NOTSET");
    for (int64_t i = 0; i < spatialRank; i++) {
      if (!needPadding) {
        flattenPaddings.push_back(pads[i].getLiteral());
        flattenPaddings.push_back(pads[i + spatialRank].getLiteral());
      } else {
        int64_t kdTerm = (kernelShape[i].getLiteral() - 1) * dilations[i] + 1;
        int64_t padFront = pads[i].getLiteral();
        int64_t padBack =
            (outputDims[i + spatialOffset].getLiteral() - 1) * strides[i] +
            kdTerm - inputShape[i + spatialOffset] - padFront;
        flattenPaddings.push_back(padFront);
        flattenPaddings.push_back(padBack);
      }
    }

    auto dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
        rewriter.getContext(), 0, 1, inputSpatialDimensions, 1, 0,
        kernelDimensions, 0, 1, outputSpatialDimensions);

    auto convResult = rewriter.create<mhlo::ConvolutionOp>(loc, outputType,
        inputOperand, filterOperand, rewriter.getI64VectorAttr(strides),
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank, 2}, rewriter.getI64Type()),
            flattenPaddings),
        DenseIntElementsAttr(), rewriter.getI64VectorAttr(dilations), nullptr,
        dimension_numbers, groupNum, 1, nullptr);

    Value result;
    if (!hasBias) {
      result = convResult;
    } else {
      Value finalB;
      Value resultShape = rewriter.create<shape::ShapeOfOp>(loc, convResult);
      finalB = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          loc, outputType, biasOperand, resultShape, rewriter.getI64TensorAttr({1}));
      result = rewriter.create<mhlo::AddOp>(loc, convResult, finalB);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXConvOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
