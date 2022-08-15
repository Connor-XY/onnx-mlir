// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s
func @test_convolution(%arg0 : tensor<1x1x5x5xf32>, %arg1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
  "func.return"(%0) : (tensor<1x1x3x3xf32>) -> ()
// CHECK-LABEL: @test_convolution
// CHECK: %1 = mhlo.convolution(%arg0, %arg1)
// CHECK-SAME: dim_numbers = [b, f, ?, ?, ?, 1]x[o, i, ?, 1]->[b, f, ?, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1, 1, 1], pad = [[0, 0], [0, 0], [0, 0], [0, 0]], rhs_dilate = [1, 1, 1, 1]} 
// CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
}

func @test_convolution_with_padding(%arg0 : tensor<1x1x5x5xf32>, %arg1 : tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", kernel_shape = [3,3], pads = [1, 1, 1, 1]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()
// CHECK-LABEL: @test_convolution
// CHECK: %1 = mhlo.convolution(%arg0, %arg1)
// CHECK-SAME: dim_numbers = [b, f, ?, ?, ?, 1]x[o, i, ?, 1]->[b, f, ?, ?, ?, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1, 1, 1], pad = [[0, 0], [0, 0], [1, 1], [1, 1]], rhs_dilate = [1, 1, 1, 1]} 
// CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32>
}