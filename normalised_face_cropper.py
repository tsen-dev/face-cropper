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

        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1)
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True)

    def crop_faces_from_image(self, image):
        """
        Crop out and return the faces in the supplied image. Any roll in the faces is reversed before cropping.
        :param image: The image to be cropped. Must be in RGB format.
        :return: A list of sub-images containing only the faces.
        """

        detected_faces = self.face_detector.process(image).detections

        if detected_faces is not None:
            normalised_face_images = []
            image_height, image_width = image.shape[:2]

            for face in detected_faces:
                face_box = face.location_data.relative_bounding_box
                face_bounds = np.ndarray.astype(np.rint(np.array([
                    [face_box.xmin * image_width, face_box.ymin * image_height],  # Bottom left
                    [(face_box.xmin + face_box.width) * image_width, (face_box.ymin + face_box.height) * image_height]])), np.int)  # Top right

                face_image = self.safe_crop_image(
                    image,
                    top=face_bounds[0][1],
                    bottom=face_bounds[1][1],
                    left=face_bounds[0][0],
                    right=face_bounds[1][0])

                detected_landmarks = self.landmark_detector.process(face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_height, face_width = face_image.shape[:2]
                    face = detected_landmarks[0]

                    left_eye_centre, right_eye_centre = self.get_left_and_right_eye_centres(
                        [face.landmark[landmark] for landmark in NormalisedFaceCropper.left_eye_landmark_indices],
                        [face.landmark[landmark] for landmark in NormalisedFaceCropper.right_eye_landmark_indices],
                        (face_width, face_height))
                    eyes_midpoint = self.get_eyes_midpoint(left_eye_centre, right_eye_centre, face_height)

                    roll_angle = self.get_face_roll_angle(left_eye_centre, right_eye_centre)
                    rotation_matrix = cv2.getRotationMatrix2D(eyes_midpoint, -roll_angle, 1)
                    rotated_landmarks = self.rotate_landmarks(
                        [face.landmark[landmark] for landmark in NormalisedFaceCropper.face_edge_landmark_indices],
                        rotation_matrix,
                        (face_width, face_height))

                    rotated_face_image = cv2.warpAffine(face_image, rotation_matrix, (image.shape[1], image.shape[0]))
                    normalised_face_images.append(
                        self.safe_crop_image(
                            rotated_face_image,
                            top=rotated_landmarks[1, 3],
                            bottom=rotated_landmarks[1, 1],
                            left=rotated_landmarks[0, 0],
                            right=rotated_landmarks[0, 2]))

            return normalised_face_images


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
        :return: A 2xn integer matrix where n is the number of landmarks. The first row contains the new x values while
        the second row contains the new y values. Each column contains the new coordinates for a landmark such that the
        first column stores the first landmark's new location, and so on.
        """

        rotated_landmarks = np.matmul(rotation_matrix, np.array(
                [[landmark.x * image_size[0] for landmark in landmarks],
                [landmark.y * image_size[1] for landmark in landmarks],
                [1, 1, 1, 1]]))

        return np.ndarray.astype(np.rint(rotated_landmarks), np.int)


    def safe_crop_image(self, image, top, bottom, left, right):
        """
        Crop the supplied image within the provided boundaries. If a boundary is outside the perimeter of the image, it
        is clipped to the perimeter edge value.
        :param image: The image to be cropped.
        :param top: Maximum y coordinate of the crop boundary. Must be a row number instead of a height value.
        :param bottom: Minimum y coordinate of the crop boundary. Must be a row number instead of a height value.
        :param left: Minimum x coordinate of the crop boundary.
        :param right: Maximum x coordinate of the crop boundary.
        :return: The cropped image.
        """

        if top < 0: top = 0
        elif top >= image.shape[0]: top = image.shape[0] - 1

        if bottom < 0: bottom = 0
        elif bottom >= image.shape[0]: bottom = image.shape[0] - 1

        if left < 0: left = 0
        elif left >= image.shape[1]: left = image.shape[1] - 1

        if right < 0: right = 0
        elif right >= image.shape[1]: right = image.shape[1] - 1

        return image[top:bottom, left:right+1]




