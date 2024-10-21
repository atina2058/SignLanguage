import numpy as np
import tensorflow as tf


class predictSign(object):
    def __init__(
        self,
        model_path='mlModel/predict/model.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        # print(type(result))
        # print(result.shape)

        print('probability = ', max(max(result)))

        # print(np.argmax(np.squeeze(result))/1)

        # print(total)
        if(max(max(result)) > 0.89):
            result_index = np.argmax(np.squeeze(result))
        else:
            result_index = 27

        # print(np.argmax(np.squeeze(result)))

        return result_index
