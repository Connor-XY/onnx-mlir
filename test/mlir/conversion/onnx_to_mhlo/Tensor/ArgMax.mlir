// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = -1 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_argmax_verifier_1(%arg0: tensor<5x5x1x32xf32>)    
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_2_]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>) -> tensor<5x5x1x32xi64>
// CHECK-NEXT:    [[VAR_4_:%.+]]:2 = mhlo.reduce(%arg0 init: [[VAR_1_]]), ([[VAR_3_]] init: [[VAR_0_]]) across dimensions = [3] : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x5x1xf32>, tensor<5x5x1xi64>)
// CHECK-NEXT:     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-NEXT:      [[VAR_6_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_7_:%.+]] = "mhlo.select"([[VAR_6_]], %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:      [[VAR_8_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_9_:%.+]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-NEXT:      [[VAR_10_:%.+]] = "mhlo.select"([[VAR_6_]], %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      [[VAR_11_:%.+]] = "mhlo.select"([[VAR_8_]], [[VAR_9_]], [[VAR_10_]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      "mhlo.return"([[VAR_7_]], [[VAR_11_]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    [[VAR_5_:%.+]] = "mhlo.reshape"([[VAR_4_]]#1) : (tensor<5x5x1xi64>) -> tensor<5x5x1x1xi64>
}

func.func @test_argmax_verifier_2(%arg0 : tensor<5x?x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = 3 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_argmax_verifier_2(%arg0: tensor<5x?x1x32xf32>)
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C5:%.+]] = arith.constant 5 : index
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = shape.shape_of %arg0 : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_3_]], [[VAR_2_]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>, tensor<4xindex>) -> tensor<5x?x1x32xi64>
// CHECK-NEXT:    [[VAR_5_:%.+]]:2 = mhlo.reduce(%arg0 init: [[VAR_1_]]), ([[VAR_4_]] init: [[VAR_0_]]) across dimensions = [3] : (tensor<5x?x1x32xf32>, tensor<5x?x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x?x1xf32>, tensor<5x?x1xi64>)
// CHECK-NEXT:     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-NEXT:      [[VAR_9_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_10_:%.+]] = "mhlo.select"([[VAR_9_]], %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:      [[VAR_11_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_12_:%.+]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-NEXT:      [[VAR_13_:%.+]] = "mhlo.select"([[VAR_9_]], %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      [[VAR_14_:%.+]] = "mhlo.select"([[VAR_11_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      "mhlo.return"([[VAR_10_]], [[VAR_14_]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    [[VAR_6_:%.+]] = tensor.dim %arg0, [[C1]] : tensor<5x?x1x32xf32>
// CHECK-NEXT:    [[VAR_7_:%.+]] = tensor.from_elements [[C5]], [[VAR_6_]], [[C1]], [[C1]] : tensor<4xindex>
// CHECK-NEXT:    [[VAR_8_:%.+]] = "mhlo.dynamic_reshape"([[VAR_5_]]#1, [[VAR_7_]]) : (tensor<5x?x1xi64>, tensor<4xindex>) -> tensor<5x?x1x1xi64>
}
