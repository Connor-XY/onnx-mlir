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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

void padVector(
    SmallVectorImpl<int64_t> &inputVector, int64_t numPad, int64_t value) {
  inputVector.insert(inputVector.begin(), numPad, value);
}

struct ONNXConvOpLoweringToMhlo
    : public ConversionPattern {
  ONNXConvOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXConvOp::getOperationName(), 1,
            ctx) {}

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

    SmallVector<int64_t> inputSpatialDimensions;
    for (int64_t i = 2; i < rank; i++) {
      inputSpatialDimensions.push_back(inputType.getDimSize(i));
    }

    SmallVector<int64_t> kernalDimensions;
    for (size_t i = 0; i < kernelShape.size(); i++) {
      kernalDimensions.push_back(kernelShape[i].getLiteral());
    }

    SmallVector<int64_t> outputSpatialDimensions;
    for (int64_t i = 2; i < outputRank; i++) {
      outputSpatialDimensions.push_back(outputDims[i].getLiteral());
    }

    // paddings
    auto pads = shapeHelper.pads;
    auto padding = convOp.auto_pad();
    int64_t spatialRank = rank - spatialOffset;
    SmallVector<int64_t> flattenPaddings;
    for (int64_t i = 0; i < 2 * spatialOffset; i++) {
      flattenPaddings.push_back(0);
    }
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

    padVector(strides, spatialOffset, 1);
    padVector(dilations, spatialOffset, 1);

    auto dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 1, inputSpatialDimensions, 1, 0, kernalDimensions, 0, 1, outputSpatialDimensions);

    auto convResult = rewriter.create<mhlo::ConvOp>(loc, outputType,
        inputOperand, filterOperand,
        rewriter.getI64VectorAttr(strides),
        DenseIntElementsAttr::get(
          RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
          flattenPaddings),
        DenseIntElementsAttr(),
        rewriter.getI64VectorAttr(dilations),
        nullptr,
        dimension_numbers, groupNum, 1, nullptr);

    Value result;
    if (!hasBias) {
      result = convResult;
    } else {
      Value finalB;
      finalB = rewriter.create<mhlo::BroadcastInDimOp>(
          loc, outputType, biasOperand, rewriter.getI64TensorAttr({0}));
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
