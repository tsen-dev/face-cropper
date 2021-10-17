import mediapipe as mp
import numpy as np
import cv2
import math


class NormalisedFaceCropper:
    NO_FACES_DETECTED = 0

    def __init__(self):
        """
        Initialises a FaceCropper object
        """

        self.face_detector = mp.solutions.face_detection.FaceDetection()
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()

    def crop_face_from_image(self, image):
        """
        Crops out and returns the face in the supplied image. If the face is rotated in the plane of the image, this is reversed before cropping
        :param image: The image to be cropped. Must be in RGB format.
        :return: A sub-image containing only the face.
        """

        # Detect face and landmarks
        detected_faces = self.face_detector.process(image).detections
        detected_landmarks = self.landmark_detector.process(image).multi_face_landmarks

        if detected_faces is None or detected_landmarks is None: return None

        else:
            image_height, image_width = image.shape[:2]
            face_landmarks = detected_landmarks[0].landmark

            left_eye_landmark_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_landmark_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

            left_eye_centre = (
            round(np.sum([face_landmarks[index].x for index in left_eye_landmark_indices]) * image_width / len(left_eye_landmark_indices)),
            image_height - 1 - round(np.sum([face_landmarks[index].y for index in left_eye_landmark_indices]) * image_height / len(left_eye_landmark_indices)))

            right_eye_centre = (
            round(np.sum([face_landmarks[index].x for index in right_eye_landmark_indices]) * image_width / len(right_eye_landmark_indices)),
            image_height - 1 - round(np.sum([face_landmarks[index].y for index in right_eye_landmark_indices]) * image_height / len(right_eye_landmark_indices)))

            im = image.copy()
            cv2.circle(im, (left_eye_centre[0], image_height - left_eye_centre[1]), 5, (0, 255, 0))
            cv2.circle(im, (right_eye_centre[0], image_height - right_eye_centre[1]), 5, (255, 255, 255))
            m = (
                int((right_eye_centre[0] + left_eye_centre[0]) / 2),
                image_height - 1 - int((right_eye_centre[1] + left_eye_centre[1]) / 2)
            )

            face = detected_faces[0].location_data.relative_bounding_box
            left, bottom, right, top = int(face.xmin * image_width), int(face.ymin * image_height), \
                                       int((face.xmin + face.width) * image_width), int(
                (face.ymin + face.height) * image_height)

            for landmark in face_landmarks:
                cv2.circle(im, (round(landmark.x*image_width), round(landmark.y*image_height)), 1, (0, 0, 255))

            cv2.rectangle(im, (left, bottom), (right, top), (0, 255, 0))
            cv2.circle(im, m, 5, (255, 0, 255))



            gradient = 0

            if right_eye_centre[1] == left_eye_centre[1]:  # 0 or 180 degree rotation
                if right_eye_centre[0] >= left_eye_centre[0]: rotation_angle = 0
                else: rotation_angle = 180

            elif right_eye_centre[0] == left_eye_centre[0]:  # 90 or 270 degree rotation
                if right_eye_centre[1] > left_eye_centre[1]: rotation_angle = 90
                else: rotation_angle = 270

            else:  # Rotation between 0 and 360 degrees excluding 0, 90, and 270
                gradient = (right_eye_centre[1] - left_eye_centre[1]) / (right_eye_centre[0] - left_eye_centre[0])
                if right_eye_centre[0] > left_eye_centre[0]: rotation_angle = np.degrees(np.arctan(gradient))  # Rotation between 0 and 90 or 270 i.e. -90 and 0 degrees
                else: rotation_angle = 180 + np.degrees(np.arctan(gradient))  # Rotation between 90 and 270 degrees

            print("left:{0}\nright:{1}\nx:{2}\ny:{3}\ngradient:{4}\nangle:{5}\n\n".format(left_eye_centre, right_eye_centre, right_eye_centre[0] - left_eye_centre[0], right_eye_centre[1] - left_eye_centre[1],gradient, rotation_angle))

            rotation_matrix = cv2.getRotationMatrix2D(m, -rotation_angle, 1)
            image_blac = cv2.warpAffine(image, rotation_matrix, image.shape[:2])

            face = detected_faces[0].location_data.relative_bounding_box
            # left, bottom, right, top = int(face.xmin * image_width), int(face.ymin * image_height), \
            #                            int((face.xmin + face.width) * image_width), int(
            #     (face.ymin + face.height) * image_height)

            left = np.matmul(rotation_matrix, np.array([face_landmarks[234].x*image_width, face_landmarks[234].y*image_height, 1]))
            bottom = np.matmul(rotation_matrix, np.array([face_landmarks[152].x * image_width, face_landmarks[152].y * image_height, 1]))
            right = np.matmul(rotation_matrix, np.array([face_landmarks[454].x * image_width, face_landmarks[454].y * image_height, 1]))
            top = np.matmul(rotation_matrix, np.array([face_landmarks[10].x * image_width, face_landmarks[10].y * image_height, 1]))

            cv2.rectangle(im, (round(left[0]), round(bottom[1])), (round(right[0]), round(top[1])), (0, 128, 128))
            cv2.imshow("masked", im)

            return image_blac[round(top[1]):round(bottom[1]), round(left[0]):round(right[0])]


    # def get_eye_centre(self, eye_landmarks_x, eye_coordinates_y, image_size):
    #     """
    #     Calculate and return the pixel coordinates of the centre of an eye in the face
    #     :param eye_landmarks: The landmarks for the eye. Must be a list of
    #     :param
    #     mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    #     :return: A tuple containing the pixel coordinates of the centre of the eye.
    #     """
    #
    #
    #
    # def get_face_rotation(self, left_eye_centre, right_eye_centre):
    #     """
    #     Calculates the in-plane rotation angle of a face using the centre coordinates of the eyes.
    #     :param left_eye_centre: Pixel coordinate of the left eye's centre.
    #     :param right_eye_centre: Pixel coordinate of the right eye's centre.
    #     :return: The in-plane rotation angle of the face in degrees.
    #     """
