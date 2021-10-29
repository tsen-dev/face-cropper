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

            for face in detected_faces:
                face_box_inflation = self.get_bounding_box_inflation_factor(face.location_data.relative_keypoints[:2], 1.5, 0.5)
                face_image = self.get_inflated_face_image(image, face.location_data.relative_bounding_box, face_box_inflation)
                detected_landmarks = self.landmark_detector.process(face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0]

                    left_eye_centre, right_eye_centre = self.get_left_and_right_eye_centres(
                        [face_landmarks.landmark[landmark] for landmark in NormalisedFaceCropper.left_eye_landmark_indices],
                        [face_landmarks.landmark[landmark] for landmark in NormalisedFaceCropper.right_eye_landmark_indices],
                        (face_image.shape[1], face_image.shape[0]))
                    eyes_midpoint = self.get_eyes_midpoint(left_eye_centre, right_eye_centre, face_image.shape[0])
                    roll_angle = self.get_face_roll_angle(left_eye_centre, right_eye_centre)

                    normalised_face_images.append(self.get_normalised_face_image(
                        face_image, [face_landmarks.landmark[landmark] for landmark in NormalisedFaceCropper.face_edge_landmark_indices], eyes_midpoint, roll_angle))

            return normalised_face_images


    def get_bounding_box_inflation_factor(self, eye_coordinates, amplification, base_inflation):
        """
        Calculate and return the factor at which the perimeter of the bounding box of a face should be inflated by. This is calculated
        as a function of the gradient of the roll of the face.
        :param eye_coordinates: [left_eye, right_eye] list containing the normalised coordinates of the eyes. Left and right
        are from the perspective of someone viewing the image, not the subject in the image itself. Must be
        a list of mediapipe.framework.formats.location_data_pb2.RelativeKeypoint objects.
        :param amplification: The calculated inflation factor will be multiplied by this value.
        :param base_inflation: A base inflation to be added to the final inflation factor. Should be a decimal representing
        a percentage e.g. 0.3 for 30% base inflation.
        :return: The inflation factor. E.g. returns 0.5 to inflate box perimeter by 50%.
        """

        gradient = (eye_coordinates[0].y - eye_coordinates[1].y) / (eye_coordinates[0].x - eye_coordinates[1].x)
        inflation_factor = np.abs(np.degrees(np.arctan(gradient))) / 90

        return (inflation_factor * amplification) + base_inflation


    def get_inflated_face_image(self, image, face_box, inflation):
        """
        Crop and return the located face in the image. The perimeter of the crop is inflated using the provided inflation factor.
        :param image: The image containing the face.
        :param face_box: The bounding box of the detected face. Must be a
        mediapipe.framework.formats.location_data_pb2.RelativeBoundingBox object.
        :param inflation: The factor at which the bounding box of a face should be inflated. E.g. 0.5 will inflate the box's
        area by 50% around the centre.
        :return: A sub-image containing only the face.
        """

        im = image.copy()
        cv2.rectangle(im, (round(face_box.xmin * image.shape[1]), round(face_box.ymin * image.shape[0])),
                      (round((face_box.xmin+face_box.width) * image.shape[1]), round((face_box.ymin+face_box.height) * image.shape[0])), (255, 0, 0))

        width_inflation, height_inflation = face_box.width * inflation, face_box.height * inflation

        inflated_face_bounds = np.ndarray.astype(np.rint(np.array([
            [(face_box.xmin - width_inflation / 2) * image.shape[1], (face_box.ymin - height_inflation / 2) * image.shape[0]],  # (x, y) of bottom left
            [(face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1], (face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0]]])),  # (x, y) of top right
            np.int)

        cv2.rectangle(im, (inflated_face_bounds[0][0], inflated_face_bounds[0][1]),
                      (inflated_face_bounds[1][0], inflated_face_bounds[1][1]), (0, 255, 0))
        cv2.imshow("im", im)

        return self.safe_crop_image(image, inflated_face_bounds[0][1], inflated_face_bounds[1][1], inflated_face_bounds[0][0], inflated_face_bounds[1][0])


    def get_left_and_right_eye_centres(self, left_eye_landmarks, right_eye_landmarks, image_size):
        """
        Calculate and return the pixel coordinates of the centres of the left and right eyes in the face. The y
        coordinate is converted from a row number to a height value so that the y coordinate increases for points higher
        up in the image, instead of decreasing. All values are rounded to the nearest integer.
        :param left_eye_landmarks: The landmarks for the left eye. Left is from the perspective of someone viewing the
        image, not the subject in the image itself. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param right_eye_landmarks: The landmarks for the right eye. Right is from the perspective of someone viewing the
        image, not the subject in the image itself. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param image_size: (width, height) tuple containing the dimensions of the image with the face.
        :return: Two [x, y] integer arrays containing the pixel coordinates of the centres of the left and right eyes respectively.
        Left and right are from the perspective of someone viewing the image, not the subject in the image itself.
        """

        left_eye_centre = np.ndarray.astype(np.rint(np.array([
            np.sum([landmark.x for landmark in left_eye_landmarks]) * image_size[0] / len(left_eye_landmarks),
            image_size[1] - 1 - np.sum([landmark.y for landmark in left_eye_landmarks]) * image_size[1] / len(left_eye_landmarks)])), np.int)

        right_eye_centre = np.ndarray.astype(np.rint(np.array([
            np.sum([landmark.x for landmark in right_eye_landmarks]) * image_size[0] / len(right_eye_landmarks),
            image_size[1] - 1 - np.sum([landmark.y for landmark in right_eye_landmarks]) * image_size[1] / len(right_eye_landmarks)])), np.int)

        return left_eye_centre, right_eye_centre


    def get_eyes_midpoint(self, left_eye_centre, right_eye_centre, image_height):
        """
        Calculate and return the coordinates of the midpoint between the two eyes of a face. The y value is converted
        from a row number to a height value so that the y coordinate increases for points higher up in the image,
        instead of decreasing. All values are rounded to the nearest integer.
        :param left_eye_centre: [x, y] array containing the pixel coordinates of the left eye's centre. Left is from the
        perspective of someone viewing the image, not the subject in the image itself.
        :param right_eye_centre: [x, y] array containing the pixel coordinates of the right eye's centre. Right is from
        the perspective of someone viewing the image, not the subject in the image itself.
        :param image_height: The height of the image containing the eyes.
        :return: [x, y] integer array containing the pixel coordinates of the midpoint between the left and right eyes.
        """

        return np.ndarray.astype(np.rint(np.array([
            (left_eye_centre[0] + right_eye_centre[0]) / 2,
            image_height - 1 - (left_eye_centre[1] + right_eye_centre[1]) / 2])), np.int)


    def get_face_roll_angle(self, left_eye_centre, right_eye_centre):
        """
        Calculate and return the in-plane rotation angle of a face using the centre coordinates of the eyes.
        :param left_eye_centre: [x, y] array containing the pixel coordinates of the left eye's centre. The y value must be a converted
        height value instead of a row number so that the y coordinate increases for points higher up in the image. Left is from the
        perspective of someone viewing the image, not the subject in the image itself.
        :param right_eye_centre: [x, y] array containing the pixel coordinates of the right eye's centre. The y value must be a converted
        height value instead of a row number so that the y coordinate increases for points higher up in the image. Right is from
        the perspective of someone viewing the image, not the subject in the image itself.
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

        return np.ndarray.astype(np.rint(np.matmul(rotation_matrix, np.array(
                [[landmark.x * image_size[0] for landmark in landmarks],
                [landmark.y * image_size[1] for landmark in landmarks],
                [1, 1, 1, 1]]))), np.int)


    def safe_crop_image(self, image, top, bottom, left, right):
        """
        Crop the supplied image within the provided boundaries. If a boundary is outside the perimeter of the image, it
        is clipped to the perimeter edge value.
        :param image: The image to be cropped.
        :param top: Maximum integer y coordinate of the crop boundary. Must be a row number instead of a height value.
        :param bottom: Minimum integer y coordinate of the crop boundary. Must be a row number instead of a height value.
        :param left: Minimum integer x coordinate of the crop boundary.
        :param right: Maximum integer x coordinate of the crop boundary.
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


    def get_normalised_face_image(self, face_image, face_edge_landmarks, eyes_midpoint, roll_angle):
        """
        Normalises the roll of the face in the face image and returns the normalised image.
        :param face_image: Image containing only the bounding box of the face.
        :param face_edge_landmarks: The landmarks of the top, bottom, left, and right edges of the face. Must be a list of
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
        :param eyes_midpoint: [x, y] array containing the pixel coordinates of the midpoint between the left and right eyes
        of the face.
        :param roll_angle: The angle at which the face is rolled.
        :return: A normalised version of the supplied face image.
        """

        rotation_matrix = cv2.getRotationMatrix2D((int(eyes_midpoint[0]), int(eyes_midpoint[1])), -roll_angle, 1)
        rotated_landmarks = self.rotate_landmarks(face_edge_landmarks, rotation_matrix, (face_image.shape[1], face_image.shape[0]))
        rotated_face_image = cv2.warpAffine(face_image, rotation_matrix, (face_image.shape[1], face_image.shape[0]))

        return self.safe_crop_image(rotated_face_image, rotated_landmarks[1, 3], rotated_landmarks[1, 1],
                                    rotated_landmarks[0, 0], rotated_landmarks[0, 2])

