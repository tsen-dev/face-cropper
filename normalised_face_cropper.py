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

        image_width, image_height = image.shape[:2]

        # Detect face and landmarks
        detected_faces = self.face_detector.process(image).detections
        detected_landmarks = self.landmark_detector.process(image).multi_face_landmarks

        if detected_faces is None or detected_landmarks is None: return None, NormalisedFaceCropper.NO_FACES_DETECTED

        face_landmarks = detected_landmarks[0].landmark

        left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        left_eye_centre = (sum([int(face_landmarks[index].x * image_width) for index in left_eye_landmarks]) // 16,
                           sum([int(face_landmarks[index].y * image_height) for index in left_eye_landmarks]) // 16)
        right_eye_centre = (sum([int(face_landmarks[index].x * image_width) for index in right_eye_landmarks]) // 16,
                            sum([int(face_landmarks[index].y * image_height) for index in right_eye_landmarks]) // 16)
        eye_centre = ((left_eye_centre[0] + right_eye_centre[0]) // 2, (left_eye_centre[1] + right_eye_centre[1]) // 2)

        gradient = (right_eye_centre[1] - eye_centre[1]) / (right_eye_centre[0] - eye_centre[0] + 1e-20)
        rotation_angle = math.degrees(np.arctan(gradient))
        rotation_matrix = cv2.getRotationMatrix2D(eye_centre, rotation_angle, 1)
        image_blac = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        face = detected_faces[0].location_data.relative_bounding_box
        left, bottom, right, top = int(face.xmin * image_width), int(face.ymin * image_height), \
                                   int((face.xmin + face.width) * image_width), int(
            (face.ymin + face.height) * image_height)
        if left < 0: left = 0
        if right > image_width: right = image_width
        if bottom < 0: bottom = 0
        if top > image_height: top = image_height

        return image_blac[bottom:top, left:right]

    def get_eye_centres(self, image):
        """
        Calculate and return the pixel coordinates of the centres of the eyes in a face
        :param image:
        :return:
        """

    def get_face_rotation(self, left_eye_centre, right_eye_centre):
        """
        Calculates the in-plane rotation angle of a face using the centre coordinates of the eyes.
        :param left_eye_centre: Pixel coordinate of the left eye's centre.
        :param right_eye_centre: Pixel coordinate of the right eye's centre.
        :return: The in-plane rotation angle of the face in degrees.
        """
