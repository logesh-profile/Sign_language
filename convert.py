import tensorflow as tf

model = tf.keras.models.load_model("Model/keras_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("Model/model.tflite", "wb") as f:
    f.write(tflite_model)

print("Done! model.tflite saved!")