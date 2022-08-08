// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

// Test normal case
func.func @test_flatten(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten
// CHECK: %0 = "mhlo.reshape"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<25x32xf32>
}

// -----

// Test when axis is negative
func.func @test_flatten_negative_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = -2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten_negative_axis
// CHECK: %0 = "mhlo.reshape"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<25x32xf32>
}

// -----

// Test when axis is not set
func.func @test_flatten_with_default_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten_with_default_axis
// CHECK: %0 = "mhlo.reshape"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<5x160xf32>
}

// -----

func.func @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten1
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<2x?x4xf32>) -> tensor<?x4xf32> {
// CHECK-DAG:    [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-NEXT:    [[VAR_0_:%.+]] = tensor.dim [[PARAM_0_]], [[C1]] : tensor<2x?x4xf32>
// CHECK-NEXT:    [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[C2]] : index
// CHECK-NEXT:    [[VAR_2_:%.+]] = tensor.from_elements [[VAR_1_]], [[C4]] : tensor<2xindex>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.dynamic_reshape"([[PARAM_0_]], [[VAR_2_]]) : (tensor<2x?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
}

// -----