import mediapipe as mp
import numpy as np
import cv2


class NormalisedFaceCropper:

    left_eye_landmark_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_landmark_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    face_edge_landmark_indices = [234, 152, 454, 10]

    def __init__(self):
        """
        Initialises a FaceCropper object
        """
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()

    def crop_faces_from_image(self, image):
        """
        Crop out and return the faces in the supplied image. Any roll in the faces is reversed before cropping.
        :param image: The image to be cropped. Must be in RGB format.
        :return: A list of sub-images containing only the faces.
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
                (image_width, image_height))
            eyes_midpoint = self.get_eyes_midpoint(left_eye_centre, right_eye_centre, image_height)

            roll_angle = self.get_face_roll_angle(left_eye_centre, right_eye_centre)
            rotation_matrix = cv2.getRotationMatrix2D(eyes_midpoint, -roll_angle, 1)
            rotated_landmarks = self.rotate_landmarks(
                [face_landmarks[landmark] for landmark in NormalisedFaceCropper.face_edge_landmark_indices],
                rotation_matrix,
                (image_width, image_height))

            # im = image.copy()
            # for landmark in face_landmarks:
            #     cv2.circle(im, (round(landmark.x*image_width), round(landmark.y*image_height)), 3, (0, 255, 0))
            # cv2.imshow("im", im)
            # print(face_landmarks[NormalisedFaceCropper.face_edge_landmark_indices[0]].x*image_width)

            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            return self.crop_image(
                rotated_image,
                top=rotated_landmarks[1, 3],
                bottom=rotated_landmarks[1, 1],
                left=rotated_landmarks[0, 0],
                right=rotated_landmarks[0, 2])


    def get_left_and_right_eye_centres(self, left_eye_landmarks, right_eye_landmarks, image_size):
        """
        Calculate and return the pixel coordinates of the centres of the left and right eyes in the face. The y
        coordinate is converted from a row number to a height value so that the y coordinate increases for points higher
        up in the image, instead of decreasing. All values are rounded to the nearest integer.
        :param left_eye_landmarks: The landmarks for the left eye. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param right_eye_landmarks: The landmarks for the right eye. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param image_size: (width, height) tuple containing the dimensions of the image with the face
        :return: Two (x, y) integer tuples containing the pixel coordinates of the centres of the left and right eyes.
        """

        left_eye_centre = (
            round(np.sum([landmark.x for landmark in left_eye_landmarks]) * image_size[0] / len(left_eye_landmarks)),
            image_size[1] - 1 - round(np.sum([landmark.y for landmark in left_eye_landmarks]) * image_size[1] / len(left_eye_landmarks)))

        right_eye_centre = (
            round(np.sum([landmark.x for landmark in right_eye_landmarks]) * image_size[0] / len(right_eye_landmarks)),
            image_size[1] - 1 - round(np.sum([landmark.y for landmark in right_eye_landmarks]) * image_size[1] / len(right_eye_landmarks)))

        return left_eye_centre, right_eye_centre


    def get_eyes_midpoint(self, left_eye_centre, right_eye_centre, image_height):
        """
        Calculate and return the coordinates of the midpoint between the two eyes of a face. The y value is converted
        from a row number to a height value so that the y coordinate increases for points higher up in the image,
        instead of decreasing. All values are rounded to the nearest integer.
        :param left_eye_centre: (x, y) tuple containing the pixel coordinates of the left eye's centre.
        :param right_eye_centre: (x, y) tuple containing the pixel coordinates of the left eye's centre.
        :param image_height: The height of the image containing the eyes.
        :return: (x, y) integer tuple containing the pixel coordinates of the midpoint between the left and right eyes.
        """

        return (round((left_eye_centre[0] + right_eye_centre[0]) / 2),
                image_height - 1 - round((left_eye_centre[1] + right_eye_centre[1]) / 2))


    def get_face_roll_angle(self, left_eye_centre, right_eye_centre):
        """
        Calculate and return the in-plane rotation angle of a face using the centre coordinates of the eyes.
        :param left_eye_centre: (x, y) tuple containing the pixel coordinates of the left eye's centre.
        :param right_eye_centre: (x, y) tuple containing the pixel coordinates of the right eye's centre.
        :return: The roll angle of the face in degrees.
        """

        if right_eye_centre[1] == left_eye_centre[1]:  # 0 or 180 degree roll
            if right_eye_centre[0] >= left_eye_centre[0]:
                return 0
            else:
                return 180

        elif right_eye_centre[0] == left_eye_centre[0]:  # 90 or 270 degree roll
            if right_eye_centre[1] > left_eye_centre[1]:
                return 90
            else:
                return 270

        else:  # Roll between 0 and 360 degrees excluding 0, 90, and 270
            gradient = (right_eye_centre[1] - left_eye_centre[1]) / (right_eye_centre[0] - left_eye_centre[0])
            if right_eye_centre[0] > left_eye_centre[0]:
                return np.degrees(np.arctan(gradient))  # Roll between 0 and 90 or 270 and 0 i.e. -90 and 0 degrees
            else:
                return 180 + np.degrees(np.arctan(gradient))  # Roll between 90 and 270 degrees


    def rotate_landmarks(self, landmarks, rotation_matrix, image_size):
        """
        Rotate the landmark points using the specified rotation matrix. The new points are rounded to the nearest
        values and converted to integer types.
        :param landmarks: The landmarks to be rotated. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param rotation_matrix: 3x2 transformation matrix for rotation around a specified point.
        :param image_size: (width, height) tuple containing the dimensions of the image containing the landmarks
        :return: A 2xn integer matrix. The first row contains the new x values while the second row contains the new y
        values. Each column contains the new coordinates for a landmark such that the first column stores the first
        landmark's new location, and so on.
        """

        rotated_landmarks = np.matmul(rotation_matrix, np.array(
                [[landmark.x * image_size[0] for landmark in landmarks],
                [landmark.y * image_size[1] for landmark in landmarks],
                [1, 1, 1, 1]]))

        return np.ndarray.astype(np.rint(rotated_landmarks), np.int)


    def crop_image(self, image, top, bottom, left, right):
        """
        Crop the supplied image within the provided boundaries. If a boundary is outside the perimeter of the image, it
        is clipped to the perimeter edge value.
        :param image: The image to be cropped.
        :param top: The top boundary of the crop.
        :param bottom: The bottom boundary of the crop.
        :param left: The left boundary of the crop.
        :param right: The right boundary of the crop.
        :return: The cropped image.
        """

        return image[top:bottom, left:right]




