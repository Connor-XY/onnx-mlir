/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToMhlo.cpp - ONNX dialects to Mhlo lowering -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Mhlo IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Transform/ONNX/Decompose.h"

using namespace mlir;

namespace onnx_mlir {

void populateONNXToMhloConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  populateLoweringONNXDecomposeOpToONNXPattern(patterns, ctx);
  // Math
  populateLoweringONNXElementwiseOpToMhloPattern(patterns, ctx);
  populateLoweringONNXGemmOpToMhloPattern(patterns, ctx);
  populateLoweringONNXReductionOpToMhloPattern(patterns, ctx);
  // Neural network
  populateLoweringONNXNormalizationOpToMhloPattern(patterns, ctx);
  populateLoweringONNXPoolingOpToMhloPattern(patterns, ctx);
  // Tensor
  populateLoweringONNXConcatOpToMhloPattern(patterns, ctx);
  populateLoweringONNXConstantOpToMhloPattern(patterns, ctx);
  populateLoweringONNXReshapeOpToMhloPattern(patterns, ctx);
}

Value getShapedZero(Location loc, ConversionPatternRewriter &rewriter,
    const ShapedType &inpType, Value &inp, const Type &resultType) {
  Value broadcastedZero;
  if (inpType.hasStaticShape())
    broadcastedZero =
        rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(inpType));
  else {
    Type elemType = inpType.getElementType();
    Value zero =
        rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elemType));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedZero = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, zero, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedZero;
}

//===----------------------------------------------------------------------===//
// Frontend to Mhlo Dialect lowering pass
//===----------------------------------------------------------------------===//

struct FrontendToMhloLoweringPass
    : public PassWrapper<FrontendToMhloLoweringPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-onnx-to-mhlo"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Mhlo dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToMhloLoweringPass() = default;
  FrontendToMhloLoweringPass(const FrontendToMhloLoweringPass &pass)
      : PassWrapper<FrontendToMhloLoweringPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void FrontendToMhloLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<mhlo::MhloDialect, func::FuncDialect,
      shape::ShapeDialect>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  target.addLegalOp<::mlir::UnrealizedConversionCastOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Define patterns.
  populateONNXToMhloConversionPattern(patterns, &getContext());

  // add illegal op
  target.addIllegalOp<ONNXSoftmaxOp>();

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToMhloPass() {
  return std::make_unique<FrontendToMhloLoweringPass>();
}

} // namespace onnx_mlir
