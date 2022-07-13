/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/DecomposeEinsum.hpp"

#include "Decompose.h"

using namespace mlir;

namespace onnx_mlir {

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  assert(origAttrs && "handle EXISTING ArrayAttr only");

  if (origAttrs.getValue()[0].dyn_cast<FloatAttr>()) {
    Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }

  if (origAttrs.getValue()[0].dyn_cast<IntegerAttr>()) {
    Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

/// Create an Scalar DenseElementsAttr from FloatAttr or IntergerAttr.
/// This is used to create an ONNXConstant of rank 0, e.g. tensor<f32>.
DenseElementsAttr createScalarDenseAttr(
    PatternRewriter &rewriter, Attribute attr) {
  if (attr.dyn_cast<FloatAttr>()) {
    Type elementType = rewriter.getF32Type();
    SmallVector<float, 1> wrapper;
    wrapper.emplace_back(attr.cast<FloatAttr>().getValueAsDouble());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }

  if (attr.dyn_cast<IntegerAttr>()) {
    Type elementType = rewriter.getIntegerType(64);
    SmallVector<int64_t, 1> wrapper;
    wrapper.emplace_back(attr.cast<IntegerAttr>().getInt());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

Value createUnitConstant(PatternRewriter &rewriter, Location loc) {
  return rewriter.create<ONNXNoneOp>(loc);
}

// Create an DenseElementsAttr of ArrayAttr.
// When ArrayAttr is Null, an empty Integer DenseElementAttr is returned
DenseElementsAttr createDenseArrayAttrOrEmpty(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  if (origAttrs)
    return createDenseArrayAttr(rewriter, origAttrs);

  Type elementType = rewriter.getIntegerType(64);
  int nElements = 0;
  SmallVector<int64_t, 4> wrapper(nElements, 0);
  for (int i = 0; i < nElements; ++i)
    wrapper[i] = i;

  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::makeArrayRef(wrapper));
}

Value createSequenceConstructOp(
    PatternRewriter &rewriter, mlir::Value seq, mlir::OperandRange inputs) {
  Type resType = seq.getType();
  Location loc = seq.getLoc();
  Value position = rewriter.create<ONNXNoneOp>(loc);

  for (auto input : inputs)
    seq = rewriter.create<ONNXSequenceInsertOp>(
        loc, resType, seq, input, position);

  return seq;
}

} // namespace onnx_mlir

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

RankedTensorType createResultType(
    Type outputType, int64_t axisValue, bool keepDims) {
  RankedTensorType outputShapeType = outputType.dyn_cast<RankedTensorType>();
  llvm::ArrayRef<int64_t> shapeVector = outputShapeType.getShape();
  int64_t rank = outputShapeType.getRank();
  SmallVector<int64_t, 4> reducedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != axisValue)
      reducedShape.push_back(shapeVector[i]);
    else if (keepDims)
      reducedShape.push_back(1);
  }
  Type elementType = outputShapeType.getElementType();
  RankedTensorType resultType =
      RankedTensorType::get(reducedShape, elementType);
  return resultType;
}

struct SoftmaxPattern : public RewritePattern {
  SoftmaxPattern(MLIRContext *context)
      : RewritePattern("onnx.Softmax", 1, context,
            {"onnx.Constant", "onnx.ReduceMax", "onnx.Sub", "onnx.Exp",
                "onnx.ReduceSum", "onnx.Div"}) {}
  LogicalResult matchAndRewrite(
      Operation *op0, PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    IntegerAttr axis;
    Operation::operand_range x(op0->getOperands());

    // Match
    ONNXSoftmaxOp castedOp0 = ::llvm::dyn_cast<ONNXSoftmaxOp>(op0);
    x = castedOp0.getODSOperands(0);
    Type inputType = castedOp0.input().getType();
    axis = op0->getAttrOfType<IntegerAttr>("axis");
    if (!axis)
      axis = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, /*isSigned=*/true), -1);
    int64_t axisValue = axis.getSInt();

    // Rewrite
    Location odsLoc = rewriter.getFusedLoc({op0->getLoc()});
    ONNXReduceMaxOp reduceMaxOp;
    {
      Value input = (*x.begin());
      IntegerAttr keepDimsAttr = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, /*isSigned=*/true), 1);
      ArrayAttr axisAttr = rewriter.getI64ArrayAttr({axis.getSInt()});
      RankedTensorType resultType = createResultType(
          inputType, axis.getSInt(), /*keepDims=*/true);
      reduceMaxOp = rewriter.create<ONNXReduceMaxOp>(
          odsLoc, resultType, input, axisAttr, keepDimsAttr);
    }
    ONNXSubOp subOp;
    {
      Value value0 = (*x.begin());
      Value value1 = *reduceMaxOp.getODSResults(0).begin();
      subOp = rewriter.create<ONNXSubOp>(odsLoc, inputType, value0, value1);
    }
    ONNXExpOp expOp;
    {
      Value value0 = *subOp.getODSResults(0).begin();
      expOp = rewriter.create<ONNXExpOp>(odsLoc, inputType, value0);
    }
    ONNXConstantOp axisOp;
    {
      axisOp = rewriter.create<ONNXConstantOp>(odsLoc, nullptr,
          /*value=*/rewriter.getI64TensorAttr({axisValue}));
    }
    ONNXReduceSumOp reduceSumOp;
    {
      RankedTensorType resultType = createResultType(inputType, axisValue, true);
      IntegerAttr keepDimsAttr = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, /*isSigned=*/true), 1);
      IntegerAttr noopWithEmptyAxes = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, /*isSigned=*/true), 0);
      reduceSumOp = rewriter.create<ONNXReduceSumOp>(odsLoc, resultType,
          /*input=*/*expOp.getODSResults(0).begin(),
          /*axis=*/axisOp, keepDimsAttr, noopWithEmptyAxes);
    }
    ONNXDivOp divOp;
    {
      Value value0 = *expOp.getODSResults(0).begin();
      Value value1 = *reduceSumOp.getODSResults(0).begin();
      divOp =
          rewriter.create<ONNXDivOp>(odsLoc, inputType, value0, value1);
    }
    rewriter.replaceOp(op0, divOp.getODSResults(0));
    return success();
  }
};

// clang-format off
#define VALID_CUSTOM_CALL_OP(cb) \
    cb(softmax, Softmax)         \
// clang-format on

#define GEN_FUNC(op, func_name)                                                             \
  constexpr const char *get##func_name##Name() { return #op; }                              \
  void func_name##AddPattern(RewritePatternSet& patterns, ConversionTarget& target) {       \
    patterns.add<func_name##Pattern>(patterns.getContext());                                \
    target.addIllegalOp<ONNX##func_name##Op>();                                             \
  }

VALID_CUSTOM_CALL_OP(GEN_FUNC)

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeONNXToONNXPass)

  DecomposeONNXToONNXPass() = default;
  DecomposeONNXToONNXPass(const DecomposeONNXToONNXPass& pass) {}

  StringRef getArgument() const override { return "decompose-onnx"; }

  StringRef getDescription() const override {
    return "Decompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  void runOnOperation() final;

  ListOption<std::string> ops{
      *this, "ops",
      llvm::cl::desc("List of ONNX operations to decompose."),
      llvm::cl::ZeroOrMore};
};

void DecomposeONNXToONNXPass::runOnOperation() {
  std::unordered_set<std::string> opsSet(this->ops.begin(), this->ops.end());
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithmeticDialect,
      func::FuncDialect>();

  std::unordered_map<std::string, std::function<void(RewritePatternSet&, ConversionTarget&)>>
    validDecomposeOpSet;

  validDecomposeOpSet.emplace(getSoftmaxName(), SoftmaxAddPattern);

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXClipV6Op>();
  target.addIllegalOp<ONNXClipV11Op>();
  target.addIllegalOp<ONNXClipV12Op>();
  target.addIllegalOp<ONNXEinsumOp>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXPadV2Op>();
  target.addIllegalOp<ONNXPadV11Op>();
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXResizeV11Op>();
  target.addIllegalOp<ONNXResizeV10Op>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXScatterOp>();
  target.addIllegalOp<ONNXSequenceConstructOp>();
  target.addIllegalOp<ONNXUpsampleOp>();
  target.addIllegalOp<ONNXUpsampleV9Op>();
  target.addIllegalOp<ONNXUpsampleV7Op>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<onnx_mlir::DecomposeEinsumPattern>(&getContext());

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

namespace onnx_mlir {

void populateLoweringONNXDecomposeOpToONNXPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<SoftmaxPattern>(ctx);
}
/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}

} // namespace onnx_mlir
