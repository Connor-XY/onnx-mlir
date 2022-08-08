// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_squeeze
// CHECK: %0 = "mhlo.reshape"(%arg0) : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
}

func.func @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_squeeze_unknown_dimensions
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<?x1x32x?x64xf32>)
// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:   [[VAR_0_:%.+]] = tensor.dim [[PARAM_0_]], [[C0]] : tensor<?x1x32x?x64xf32>
// CHECK-NEXT:   [[VAR_1_:%.+]] = tensor.from_elements [[VAR_0_]], [[C32]], [[C64]] : tensor<3xindex>
// CHECK-NEXT:   [[VAR_2_:%.+]] = "mhlo.dynamic_reshape"([[PARAM_0_]], [[VAR_1_]]) : (tensor<?x1x32x?x64xf32>, tensor<3xindex>) -> tensor<?x32x64xf32>
}