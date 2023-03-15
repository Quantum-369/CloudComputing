import numpy as np
import cv2
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="converted_model_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the input image
img = cv2.imread("pothole1.jpg")

# Preprocess the image
input_data = cv2.resize(img, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
input_data = input_data.astype(np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)
input_data = (input_data * 255).astype(np.uint8)

# Run inference on the TensorFlow Lite model
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the model's prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
#output_data = np.clip(output_data, 0, 1)



# Convert the prediction to binary mask and count the number of potholes
mask = (output_data > 0.99).astype(np.uint8)
potholes = np.count_nonzero(mask)

print("Number of potholes:", potholes)
print(output_data)
# import json
#
# coords = []
# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         if mask[i][j].any() == 1:
#             coords.append([i, j])
# img = cv2.imread("Image.jpg")
# for coord in coords:
#     x, y = coord
#     cv2.rectangle(img, (y - 10, x - 10), (y + 10, x + 10), (0, 255, 0), 2)
#
#
# cv2.imshow("Potholes", img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

# escape key waiting


#print(json.dumps({"potholes": coords}))

