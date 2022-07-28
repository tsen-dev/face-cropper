import mediapipe as mp
import numpy as np
import cv2


def _get_bounding_box_inflation_factor(eye_coordinates, amplification=2, base_inflation=1):
    """
    Calculate and return the factor at which the perimeter of the bounding box of a face should be inflated by. This is calculated
    as a function of the gradient of the line going through the left and right eyes.
    :param eye_coordinates: [right_eye, left_eye] list containing the normalised coordinates of the eyes. Must be
    a list of mediapipe.framework.formats.location_data_pb2.RelativeKeypoint objects.
    :param amplification: The calculated inflation factor will be multiplied by this value. Defaults to 2.
    :param base_inflation: A base inflation to be added to the final inflation factor. defaults to 1 for 100% base inflation.
    :return: The inflation factor. E.g. returns 0.5 to inflate box perimeter by 50%.
    """

    roll_angle = _get_face_roll_angle([eye_coordinates[1].x, 1 - eye_coordinates[1].y], [eye_coordinates[0].x, 1 - eye_coordinates[0].y])
    inflation_factor = np.abs(roll_angle) / 90

    return base_inflation + (inflation_factor * amplification)


def _inflate_face_image(image, face_box, inflation):
    """
    Crop and return the located face in the image. The perimeter of the crop is inflated around the center using the provided inflation factor.
    :param image: The image containing the face.
    :param face_box: The bounding box of the detected face. Must be a mediapipe.framework.formats.location_data_pb2.RelativeBoundingBox object.
    :param inflation: The factor by which the bounding box of a face should be inflated. E.g. 0.5 will inflate the box's perimeter by 50% around the centre.
    :return: A sub-image containing only the face.
    """

    width_inflation, height_inflation = face_box.width * inflation, face_box.height * inflation

    return _crop_within_bounds(
        image,
        round((face_box.ymin - height_inflation / 2) * image.shape[0]),                    # top
        round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0]),  # bottom
        round((face_box.xmin - width_inflation / 2) * image.shape[1]),                     # left
        round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1])     # right
    )


def _get_left_and_right_eye_centres(left_eye_landmarks, right_eye_landmarks):
    """
    Calculate and return the normalised coordinates of the centres of the left and right eyes in the face. The y
    coordinate is converted from a row number to a height value so that the y coordinate increases for points higher up in the image.
    :param left_eye_landmarks: The landmarks for the left eye. Must be a list of
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :param right_eye_landmarks: The landmarks for the right eye. Must be a list of
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :return: Two [x, y] arrays containing the normalised coordinates of the centres of the left and right eyes respectively.
    """

    left_eye_centre = np.array([
        np.sum([landmark.x for landmark in left_eye_landmarks]) / len(left_eye_landmarks),
        1 - np.sum([landmark.y for landmark in left_eye_landmarks]) / len(left_eye_landmarks)])

    right_eye_centre = np.array([
        np.sum([landmark.x for landmark in right_eye_landmarks]) / len(right_eye_landmarks),
        1 - np.sum([landmark.y for landmark in right_eye_landmarks]) / len(right_eye_landmarks)])

    return left_eye_centre, right_eye_centre


def _get_eyes_midpoint(left_eye_centre, right_eye_centre, image_size):
    """
    Calculate and return the pixel coordinates of the midpoint between the two eyes of a face. All values are rounded to the nearest integer.
    :param left_eye_centre: [x, y] array containing the normalised coordinates of the left eye's centre. The y
    coordinate must be a height value such that the y coordinate increases for points higher up in the image.
    :param right_eye_centre: [x, y] array containing the normalised coordinates of the right eye's centre. The y
    coordinate must be a height value such that the y coordinate increases for points higher up in the image.
    :param image_size: (height, width) tuple containing the dimensions of the image with the face.
    :return: [x, y] integer array containing the pixel coordinates of the midpoint between the left and right eyes.
    """

    return np.ndarray.astype(np.rint(np.array([
        (left_eye_centre[0] + right_eye_centre[0]) * image_size[1] / 2,
        (left_eye_centre[1] + right_eye_centre[1]) * image_size[0] / 2])), np.int)


def _get_face_roll_angle(left_eye_centre, right_eye_centre):
    """
    Calculate and return the in-plane rotation angle of a face (in degrees) using the centre coordinates of the eyes.
    (Assumes the image has been flipped i.e. the left eye is on the right of the image and vice versa)
    :param left_eye_centre: [x, y] array containing the coordinates of the left eye's centre. The y value must be a
    height value such that it is higher for points higher up in the image.
    :param right_eye_centre: [x, y] array containing the pixel coordinates of the right eye's centre. The y value must be a
    height value such that it is higher for points higher up in the image.
    :return: The roll angle of the face in degrees (Angles with magnitude 90 or below are given as anticlockwise: +ve, clockwise: -ve. Larger angles are only given as clockwise: +ve).
    """

    if right_eye_centre[1] == left_eye_centre[1]:  # 0 or 180 degree roll
        if left_eye_centre[0] >= right_eye_centre[0]:
            return 0
        else:
            return 180

    elif right_eye_centre[0] == left_eye_centre[0]:  # 90 or 270 (-90) degree roll
        if left_eye_centre[1] > right_eye_centre[1]:
            return 90
        else:
            return -90

    else:  # Roll between 0 and 360 degrees excluding 0, 90, and 270
        gradient = (left_eye_centre[1] - right_eye_centre[1]) / (left_eye_centre[0] - right_eye_centre[0])
        if left_eye_centre[0] > right_eye_centre[0]:
            return np.degrees(np.arctan(gradient))  # Roll between 0 and 90 or 270 (-90) and 0
        else:
            return 180 + np.degrees(np.arctan(gradient))  # Roll between 90 and 270 (-90) degrees


def _rotate_landmarks(landmarks, rotation_matrix, image_size):
    """
    Rotate the landmark points using the specified rotation matrix. The new points are rounded to the nearest
    values and converted to integer types.
    :param landmarks: The landmarks to be rotated. Must be a list of
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmark objects.
    :param rotation_matrix: 3x2 transformation matrix for rotation around a specified point, by a specified angle.
    :param image_size: (height, width) tuple containing the dimensions of the image containing the landmarks.
    :return: A 2xn integer matrix where n is the number of landmarks. The first row contains the new x values while
    the second row contains the new y values. Each column contains the new coordinates for a landmark such that the
    first column stores the first landmark's new location, and so on.
    """

    return np.ndarray.astype(np.rint(np.matmul(rotation_matrix, np.array([
            np.multiply([landmark.x for landmark in landmarks], image_size[1]),
            np.multiply([landmark.y for landmark in landmarks], image_size[0]),
            np.ones(len(landmarks))]))), np.int)


def _crop_within_bounds(image, top, bottom, left, right):
    """
    Crop the supplied image within the provided boundaries. If a boundary is outside the perimeter of the image, it
    is clipped to the perimeter edge value. All edge parameters are inclusive
    :param image: The image to be cropped.
    :param top: Maximum integer y coordinate of the crop boundary. Must be a row number instead of a height value such
    that it is lower for points higher up in the image.
    :param bottom: Minimum integer y coordinate of the crop boundary. Must be a row number instead of a height value such
    that it is lower for points higher up in the image.
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

    return image[top:bottom+1, left:right+1]


class FaceCropper:
    """
    Implements the following pipeline for cropping out faces from an image:
        1. Retrieves bounding boxes and approximate eye coordinates for faces in the image, using the mp.solutions.face_detection.FaceDetection network
            - For more about this network, visit https://solutions.mediapipe.dev/face_detection
        2. For faces with in-plane rotation, the FaceDetection network gives bounding boxes that are much smaller than the face, hence:
            2.1 The in-plane rotation of a face is found by calculating the angle between the line going through the eye coordinates and the horizontal
            2.2 The bounding boxes are inflated by a factor relative to the in-plane rotation of the faces, to ensure bounding boxes include the full face under rotation
        3. The inflated bounding boxes are cropped from the image and passed to the mp.solutions.face_mesh.FaceMesh network to retrieve precise eye coordinates
            3.1 The retrieved eye coordinates are used to calculate the in-plane rotation angle (like in step 2.1).
            3.2 The face image (and the landmarks) is rotated by this angle in the opposite direction, about the midpoint between the eyes,
                to correct the in-plane rotation (i.e. normalise the face)
                - The eye coordinates from the mp.solutions.face_detection.FaceDetection network in step 1 are not used for this purpose as they are not
                  precise enough and are only good enough for getting a rough idea of the rotation in the face
                - mp.solutions.face_mesh.FaceMesh has an integrated mp.solutions.face_detection.FaceDetection model. However, since the integrated network is configured to only
                  run the short-range model (see https://solutions.mediapipe.dev/face_detection#model_selection), it can't detect faces further than 2 metres away. Having a
                  separate mp.solutions.face_detection.FaceDetection model makes the face detection network selection configurable depending on requirements
            - For more about this network, visit https://solutions.mediapipe.dev/face_mesh
        4. The normalised image is cropped within a minimum bounding box containing the landmark coordinates and returned
    """

    # face_detector_model_selection values
    SHORT_RANGE = 0
    LONG_RANGE = 1

    # landmark_detector_static_image_mode values
    STATIC_MODE = True
    TRACKING_MODE = False

    # The indexes at which the relevant landmark data is stored on the mp.solutions.face_mesh.FaceMesh model's output
    _LEFT_EYE_LANDMARK_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    _RIGHT_EYE_LANDMARK_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def __init__(self, min_face_detector_confidence=0.5, face_detector_model_selection=LONG_RANGE,
                 landmark_detector_static_image_mode=STATIC_MODE, min_landmark_detector_confidence=0.5):
        """
        Initialise a FaceCropper object.
        :param min_face_detector_confidence:
        From mp.solutions.face_detection.FaceDetection documentation:
        "min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
        See details in https://solutions.mediapipe.dev/face_detection#min_detection_confidence". Defaults to 0.5.
        :param face_detector_model_selection:
        From mp.solutions.face_detection.FaceDetection documentation:
        "0 (FaceCropper.SHORT_RANGE) or 1 (FaceCropper.LONG_RANGE). 0 to select a short-range model
        that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters.
        See details in https://solutions.mediapipe.dev/face_detection#model_selection".
        1 works well as a general purpose model that detects both close and long range faces, whereas 0 is better for detecting
        close range faces with higher yaw, pitch, or 90+ degree roll. Defaults to 1.
        :param landmark_detector_static_image_mode:
        From mp.solutions.face_mesh.FaceMesh documentation:
        "Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/face_mesh#static_image_mode". Set this to False (FaceCropper.TRACKING_MODE) if the images passed to the detector are
        from the same sequence, and there is always the same one face in the sequence. Defaults to True (FaceCropper.STATIC_MODE).
        :param min_landmark_detector_confidence:
        From mp.solutions.face_mesh.FaceMesh documentation:
        "Minimum confidence value ([0.0, 1.0]) for the face landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence". Defaults to 0.5.
        """

        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_face_detector_confidence,
                                                                       model_selection=face_detector_model_selection)

        self.landmark_detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                                 static_image_mode=landmark_detector_static_image_mode,
                                                                 min_detection_confidence=min_landmark_detector_confidence)


    def get_normalised_faces(self, image):
        """
        Crop out and normalise each detected face in the specified image and return a list of face images.
        Any roll in the faces is corrected before cropping.
        :param image: A numpy.ndarray RGB image containing faces to be cropped
        :return: A list of numpy.ndarray RGB sub-images containing only the normalised faces.
        """

        normalised_face_images = []
        detected_faces = self.face_detector.process(image).detections
        if detected_faces is None: return normalised_face_images

        for face in detected_faces:
            # The mp.solutions.face_detection.FaceDetection network may rarely 'find' a face completely outside the image, so ignore those
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:
                inflation_factor = _get_bounding_box_inflation_factor(face.location_data.relative_keypoints[:2])
                face_image = _inflate_face_image(image, face.location_data.relative_bounding_box, inflation_factor)
                detected_landmarks = self.landmark_detector.process(face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0].landmark

                    left_eye_centre, right_eye_centre = _get_left_and_right_eye_centres(
                        [face_landmarks[landmark] for landmark in FaceCropper._LEFT_EYE_LANDMARK_INDICES],
                        [face_landmarks[landmark] for landmark in FaceCropper._RIGHT_EYE_LANDMARK_INDICES])
                    eyes_midpoint = _get_eyes_midpoint(left_eye_centre, right_eye_centre, face_image.shape)
                    roll_angle = _get_face_roll_angle(left_eye_centre, right_eye_centre)

                    rotation_matrix = cv2.getRotationMatrix2D((np.int(eyes_midpoint[0]), np.int(eyes_midpoint[1])), -roll_angle, 1)
                    rotated_landmarks = _rotate_landmarks(face_landmarks, rotation_matrix, face_image.shape)
                    normalised_face_image = cv2.warpAffine(face_image, rotation_matrix, (face_image.shape[1], face_image.shape[0]))

                    normalised_face_images.append(
                        _crop_within_bounds(
                            normalised_face_image,
                            rotated_landmarks[1, np.argmin(rotated_landmarks[1, :])], rotated_landmarks[1, np.argmax(rotated_landmarks[1, :])],
                            rotated_landmarks[0, np.argmin(rotated_landmarks[0, :])], rotated_landmarks[0, np.argmax(rotated_landmarks[0, :])]
                        )
                    )

        return normalised_face_images