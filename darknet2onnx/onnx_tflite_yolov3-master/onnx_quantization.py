import onnx

from quantize import quantize, QuantizationMode

# Load the onnx model
model = onnx.load('weights/export.onnx')
# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# Save the quantized model
onnx.save(quantized_model, 'weights/quantized_model.onnx')