import mediapipe as mp
import numpy as np
import cv2


class NormalisedFaceCropper:

    left_eye_landmark_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_landmark_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self):
        """
        Initialises a FaceCropper object
        """
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()

    def crop_face_from_image(self, image):
        """
        Crops out and returns the face in the supplied image. If the face is rotated in the plane of the image, this is reversed before cropping
        :param image: The image to be cropped. Must be in RGB format.
        :return: A sub-image containing only the face.
        """

        detected_landmarks = self.landmark_detector.process(image).multi_face_landmarks

        if detected_landmarks is None:  # No faces detected
            return None

        else:
            image_height, image_width = image.shape[:2]
            face_landmarks = detected_landmarks[0].landmark

            left_eye_centre, right_eye_centre = self.get_left_and_right_eye_centres(
                [face_landmarks[landmark] for landmark in NormalisedFaceCropper.left_eye_landmark_indices],
                [face_landmarks[landmark] for landmark in NormalisedFaceCropper.right_eye_landmark_indices],
                (image_width, image_height)
            )

            im = image.copy()
            cv2.circle(im, (left_eye_centre[0], image_height - left_eye_centre[1]), 5, (0, 255, 0))
            cv2.circle(im, (right_eye_centre[0], image_height - right_eye_centre[1]), 5, (255, 255, 255))
            m = (
                int((right_eye_centre[0] + left_eye_centre[0]) / 2),
                image_height - 1 - int((right_eye_centre[1] + left_eye_centre[1]) / 2)
            )

            for landmark in face_landmarks:
                cv2.circle(im, (round(landmark.x*image_width), round(landmark.y*image_height)), 1, (0, 0, 255))

            rotation_angle = self.get_in_plane_rotation_angle(left_eye_centre, right_eye_centre)

            rotation_matrix = cv2.getRotationMatrix2D(m, -rotation_angle, 1)
            image_blac = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            cv2.circle(im, (round(face_landmarks[234].x*image_width), round(face_landmarks[234].y*image_height)), 5, (0, 255, 255))
            cv2.circle(im, (round(face_landmarks[152].x*image_width), round(face_landmarks[152].y*image_height)), 5, (0, 255, 255))
            cv2.circle(im, (round(face_landmarks[454].x*image_width), round(face_landmarks[454].y*image_height)), 5, (0, 255, 255))
            cv2.circle(im, (round(face_landmarks[10].x*image_width), round(face_landmarks[10].y*image_height)), 5, (0, 255, 255))

            left = np.matmul(rotation_matrix, np.array([face_landmarks[234].x*image_width, face_landmarks[234].y*image_height, 1]))
            bottom = np.matmul(rotation_matrix, np.array([face_landmarks[152].x * image_width, face_landmarks[152].y * image_height, 1]))
            right = np.matmul(rotation_matrix, np.array([face_landmarks[454].x * image_width, face_landmarks[454].y * image_height, 1]))
            top = np.matmul(rotation_matrix, np.array([face_landmarks[10].x * image_width, face_landmarks[10].y * image_height, 1]))

            cv2.circle(im, (round(left[0]), round(left[1])), 5, (0, 255, 255))
            cv2.circle(im, (round(right[0]), round(right[1])), 5, (0, 255, 255))
            cv2.circle(im, (round(top[0]), round(top[1])), 5, (0, 255, 255))
            cv2.circle(im, (round(bottom[0]), round(bottom[1])), 5, (0, 255, 255))
            cv2.rectangle(im, (round(face_landmarks[234].x*image_width), round(face_landmarks[152].y*image_height)), (round(face_landmarks[454].x*image_width), round(face_landmarks[10].y*image_height)), (0, 255, 255))

            imrot = image_blac.copy()
            cv2.rectangle(im, (round(left[0]), round(bottom[1])), (round(right[0]), round(top[1])), (255, 0, 0))
            cv2.circle(im, (round(left[0]), round(left[1])), 5, (255, 0, 0))
            cv2.circle(im, (round(right[0]), round(right[1])), 5, (255, 0, 0))
            cv2.circle(im, (round(top[0]), round(top[1])), 5, (255, 0, 0))
            cv2.circle(im, (round(bottom[0]), round(bottom[1])), 5, (255, 0, 0))
            cv2.imshow("imrot", imrot)
            cv2.imshow("masked", im)

            return image_blac[round(top[1]):round(bottom[1]), round(left[0]):round(right[0])]


    def get_left_and_right_eye_centres(self, left_eye_landmarks, right_eye_landmarks, image_size):
        """
        Calculate and return the pixel coordinates of the centres of the left and right eyes in the face. The y
        coordinate is converted from a row number to a height value so that the y coordinate increases for points higher
        up in the image, instead of decreasing.
        :param left_eye_landmarks: The landmarks for the left eye. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param right_eye_landmarks: The landmarks for the right eye. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param image_size: (width, height) tuple containing the dimensions of the image with the face
        :return: Two (x, y) tuples containing the pixel coordinates of the centres of the left and right eyes.
        """

        left_eye_centre = (
            round(np.sum([landmark.x for landmark in left_eye_landmarks]) * image_size[0] / len(left_eye_landmarks)),
            image_size[1] - 1 - round(np.sum([landmark.y for landmark in left_eye_landmarks]) * image_size[1] / len(left_eye_landmarks)))

        right_eye_centre = (
            round(np.sum([landmark.x for landmark in right_eye_landmarks]) * image_size[0] / len(right_eye_landmarks)),
            image_size[1] - 1 - round(np.sum([landmark.y for landmark in right_eye_landmarks]) * image_size[1] / len(right_eye_landmarks)))

        return left_eye_centre, right_eye_centre

    def get_in_plane_rotation_angle(self, left_eye_centre, right_eye_centre):
        """
        Calculate and return the in-plane rotation angle of a face using the centre coordinates of the eyes.
        :param left_eye_centre: (x, y) tuple containing the pixel coordinates of the left eye's centre.
        :param right_eye_centre: (x, y) tuple containing the pixel coordinates of the right eye's centre.
        :return: The in-plane rotation angle of the face in degrees.
        """

        if right_eye_centre[1] == left_eye_centre[1]:  # 0 or 180 degree rotation
            if right_eye_centre[0] >= left_eye_centre[0]:
                return 0
            else:
                return 180

        elif right_eye_centre[0] == left_eye_centre[0]:  # 90 or 270 degree rotation
            if right_eye_centre[1] > left_eye_centre[1]:
                return 90
            else:
                return 270

        else:  # Rotation between 0 and 360 degrees excluding 0, 90, and 270
            gradient = (right_eye_centre[1] - left_eye_centre[1]) / (right_eye_centre[0] - left_eye_centre[0])
            if right_eye_centre[0] > left_eye_centre[0]:
                return np.degrees(np.arctan(gradient))  # Rotation between 0 and 90 or 270 and 0 i.e. -90 and 0 degrees
            else:
                return 180 + np.degrees(np.arctan(gradient))  # Rotation between 90 and 270 degrees

