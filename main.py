import csv
import copy
import cv2 as cv
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark, calc_bounding_rect,draw_sentence
import mediapipe as mp
from app_files import get_args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    consecutive_word = list()
    to_display_sentence = list()
    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if key == 8:
            if len(to_display_sentence) >= 1:
                to_display_sentence.pop()
            print("backspace pressed")
        if key == 0:
            while len(to_display_sentence) >= 1:
                to_display_sentence.pop()
            print("delete pressed")
        # print(key)
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        # print(debug_image.shape)
        # cv.imshow("debug_image",debug_image)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        debug_image = draw_sentence(debug_image, to_display_sentence)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                hand_sign_id, probab = keypoint_classifier(pre_processed_landmark_list, handedness.classification[0].label[0:])
                pr = float('%.3f' % float(probab))
                p = str(pr)
                # print(type(p))
                debug_image = draw_landmarks(debug_image, landmark_list)
                flag = 0
                # debug_image = draw_info_text(
                #     debug_image,
                #     handedness,
                #     keypoint_classifier_labels[hand_sign_id], p, flag)
                flag = 1
                debug_image = draw_info_text(brect,
                                             debug_image,
                                             handedness,
                                             keypoint_classifier_labels[hand_sign_id], p, flag)
                debug_image = draw_sentence(debug_image, to_display_sentence)
                if probab > 0.5:
                    consecutive_word.append(keypoint_classifier_labels[hand_sign_id])

                    # Python program to check if all
                    # elements in a List are same
                    if len(consecutive_word) > 30:
                        def chkList(lst):
                            return len(set(lst)) == 1

                        lst = consecutive_word[-30:]
                        if chkList(lst):
                            if len(to_display_sentence) == 0:
                                to_display_sentence.append(lst[-1])

                            if to_display_sentence[-1] != lst[-1]:
                                to_display_sentence.append(lst[-1])
                            #     print(to_display_sentence)
                        else:
                            lst = []
                    # debug_image = draw_sentence(debug_image, to_display_sentence)
                    print(to_display_sentence)

        cv.imshow('Gesture Recognition', debug_image)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
