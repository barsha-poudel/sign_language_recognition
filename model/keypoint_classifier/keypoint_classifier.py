import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
            self,
            model_path1='model/keypoint_classifier/keypoint_classifier.tflite',
            model_path2='model/keypoint_classifier/keypoint_classifier.tflite',
            num_threads=1,
    ):
        self.interpreter1 = tf.lite.Interpreter(model_path=model_path1,
                                               num_threads=num_threads)

        self.interpreter1.allocate_tensors()
        self.input_details1 = self.interpreter1.get_input_details()
        self.output_details1 = self.interpreter1.get_output_details()

        self.interpreter2 = tf.lite.Interpreter(model_path=model_path2,
                                                num_threads=num_threads)

        self.interpreter2.allocate_tensors()
        self.input_details2 = self.interpreter2.get_input_details()
        self.output_details2 = self.interpreter2.get_output_details()

    def __call__(
            self,
            landmark_list,
            hand_choice
    ):
        if hand_choice == "Left":
            input_details_tensor_index = self.input_details1[0]['index']
            self.interpreter1.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter1.invoke()

            output_details_tensor_index = self.output_details1[0]['index']

            result = self.interpreter1.get_tensor(output_details_tensor_index)
            # for res in result:
            #     for r in res:
            #         print(float(r)*100.0000)
            #
            result_index = np.argmax(np.squeeze(result))
            probab = np.squeeze(result)[result_index]
            # print("Left tensor")
            return result_index, probab

        input_details_tensor_index = self.input_details2[0]['index']
        self.interpreter2.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter2.invoke()

        output_details_tensor_index = self.output_details2[0]['index']

        result = self.interpreter2.get_tensor(output_details_tensor_index)
        # for res in result:
        #     for r in res:
        #         print(float(r)*100.0000)
        #
        result_index = np.argmax(np.squeeze(result))
        probab = np.squeeze(result)[result_index]

        # print("---------------------------------------------------------------")
        return result_index, probab
