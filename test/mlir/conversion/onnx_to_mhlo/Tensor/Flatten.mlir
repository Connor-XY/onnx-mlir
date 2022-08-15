// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

// Test normal case
func @test_flatten(%arg0 : tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32>
  "func.return"(%0) : (tensor<5x5x32xf32>) -> ()
// CHECK-LABEL:  func @test_flatten
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32> {
// CHECK-DAG:    [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NEXT:   [[VAR_0_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c0_]] : tensor<5x5x1x32xf32>
// CHECK-DAG:    [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-NEXT:   [[VAR_1_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c1_]] : tensor<5x5x1x32xf32>
// CHECK-DAG:    [[VAR_c1_0_:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-NEXT:   [[VAR_2_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c2_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_3_:%.+]] = arith.muli [[VAR_c1_0_]], [[VAR_2_]] : index
// CHECK-DAG:    [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-NEXT:   [[VAR_4_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c3_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = arith.muli [[VAR_3_]], [[VAR_4_]] : index
// CHECK-NEXT:   [[VAR_6_:%.+]] = tensor.from_elements [[VAR_0_]], [[VAR_1_]], [[VAR_5_]] : tensor<3xindex>
// CHECK-NEXT:   [[VAR_7_:%.+]] = "mhlo.dynamic_reshape"([[PARAM_0_]], [[VAR_6_]]) : (tensor<5x5x1x32xf32>, tensor<3xindex>) -> tensor<5x5x32xf32>
// CHECK-NEXT:   return [[VAR_7_]] : tensor<5x5x32xf32>
// CHECK-NEXT:   }
}

// -----

// Test when axis is negative
func @test_flatten_negative_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = -2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32>
  "func.return"(%0) : (tensor<5x5x32xf32>) -> ()
// CHECK-LABEL:  func @test_flatten_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x5x32xf32> {
// CHECK-DAG:    [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NEXT:   [[VAR_0_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c0_]] : tensor<5x5x1x32xf32>
// CHECK-DAG:    [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-NEXT:   [[VAR_1_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c1_]] : tensor<5x5x1x32xf32>
// CHECK-DAG:    [[VAR_c1_0_:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-NEXT:   [[VAR_2_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c2_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_3_:%.+]] = arith.muli [[VAR_c1_0_]], [[VAR_2_]] : index
// CHECK-DAG:    [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-NEXT:   [[VAR_4_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c3_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = arith.muli [[VAR_3_]], [[VAR_4_]] : index
// CHECK-NEXT:   [[VAR_6_:%.+]] = tensor.from_elements [[VAR_0_]], [[VAR_1_]], [[VAR_5_]] : tensor<3xindex>
// CHECK-NEXT:   [[VAR_7_:%.+]] = "mhlo.dynamic_reshape"([[PARAM_0_]], [[VAR_6_]]) : (tensor<5x5x1x32xf32>, tensor<3xindex>) -> tensor<5x5x32xf32>
// CHECK-NEXT:   return [[VAR_7_]] : tensor<5x5x32xf32>
// CHECK-NEXT:   }
}

// -----

// Test when axis is not set
func @test_flatten_with_default_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<5x160xf32> {
  %0 = "onnx.Flatten"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<5x160xf32>
  "func.return"(%0) : (tensor<5x160xf32>) -> ()
// CHECK-LABEL:  func @test_flatten_with_default_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x160xf32> {
// CHECK-DAG:    [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NEXT:   [[VAR_0_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c0_]] : tensor<5x5x1x32xf32>
// CHECK-DAG:    [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[VAR_c1_0_:%.+]] = arith.constant 1 : index
// CHECK-NEXT:   [[VAR_1_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c1_0_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_2_:%.+]] = arith.muli [[VAR_c1_]], [[VAR_1_]] : index
// CHECK-DAG:    [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-NEXT:   [[VAR_3_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c2_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_4_:%.+]] = arith.muli [[VAR_2_]], [[VAR_3_]] : index
// CHECK-DAG:    [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-NEXT:   [[VAR_5_:%.+]] = tensor.dim [[PARAM_0_]], [[VAR_c3_]] : tensor<5x5x1x32xf32>
// CHECK-NEXT:   [[VAR_6_:%.+]] = arith.muli [[VAR_4_]], [[VAR_5_]] : index
// CHECK-NEXT:   [[VAR_7_:%.+]] = tensor.from_elements [[VAR_0_]], [[VAR_6_]] : tensor<2xindex>
// CHECK-NEXT:   [[VAR_8_:%.+]] = "mhlo.dynamic_reshape"([[PARAM_0_]], [[VAR_7_]]) : (tensor<5x5x1x32xf32>, tensor<2xindex>) -> tensor<5x160xf32>
// CHECK-NEXT:   return [[VAR_8_]] : tensor<5x160xf32>
// CHECK-NEXT:   }
}
