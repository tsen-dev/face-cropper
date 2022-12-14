import unittest
import face_cropper
import numpy as np
import cv2


class TestFaceCropper(unittest.TestCase):

    class FaceBox:
        def __init__(self, xmin, width, ymin, height):
            self.xmin = xmin
            self.width = width
            self.ymin = ymin
            self.height = height


    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y


    def test__get_face_roll_angle(self):
        self.assertEqual(face_cropper._get_face_roll_angle([1, 0], [0, 0]), 0)
        self.assertEqual(face_cropper._get_face_roll_angle([0, 1], [0, 0]), 90)
        self.assertEqual(face_cropper._get_face_roll_angle([0, 0], [0, 1]), -90)
        self.assertEqual(face_cropper._get_face_roll_angle([0, 0], [1, 0]), 180)

        self.assertEqual(face_cropper._get_face_roll_angle([1, 1], [0, 0]), 45)
        self.assertEqual(face_cropper._get_face_roll_angle([0, 1], [1, 0]), 135)
        self.assertEqual(face_cropper._get_face_roll_angle([0, 0], [1, 1]), 225)
        self.assertEqual(face_cropper._get_face_roll_angle([1, 0], [0, 1]), -45)


    def test__crop_within_bounds(self):
        image = np.array([i for i in range(50 * 100)]).reshape((50, 100))

        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, 0, 49, 0, 99), image), True)
        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, -50, 200, -50, 200), image), True)
        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, 25, 75, 50, 150), image[25:, 50:]), True)
        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, 25, 75, -50, 50), image[25:, :50 + 1]), True)
        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, -25, 25, -50, 50), image[:25 + 1, :50 + 1]), True)
        self.assertEqual(np.array_equal(face_cropper._crop_within_bounds(image, -25, 25, 50, 150), image[:25 + 1, 50:]), True)


    def test__get_inflated_face_image(self):
        image = np.array([i for i in range(50 * 100)]).reshape((50, 100))

        self.assertEqual(np.array_equal(face_cropper._get_inflated_face_image(image, TestFaceCropper.FaceBox(0.50, 0.25, 0.50, 0.25), 0), image[25:38 + 1, 50:75 + 1]), True)
        self.assertEqual(np.array_equal(face_cropper._get_inflated_face_image(image, TestFaceCropper.FaceBox(0.50, 0.25, 0.50, 0.25), 1), image[19:44 + 1, 38:88 + 1]), True)


    def test__get_segmented_face_image(self):
        image = np.array([i for i in range(200 * 100)], dtype=np.uint8).reshape((200, 100))
        result = np.empty(image.shape, dtype=np.uint8)
        for i, row in enumerate(result):
            for j, col in enumerate(row):
                if 50 <= i <= 150 and 25 <= j <= 75:
                    result[i, j] = image[i, j]
                else:
                    result[i, j] = 0

        self.assertEqual(
            np.array_equal(
                face_cropper._get_segmented_face_image(
                    image,
                    [(0, 1, 2), (3, 2, 1)],
                    [TestFaceCropper.Landmark(0.25, 0.25), TestFaceCropper.Landmark(0.75, 0.25), TestFaceCropper.Landmark(0.25, 0.75), TestFaceCropper.Landmark(0.75, 0.75)]),
                result
            ),
            True
        )


    def test__get_left_and_right_eye_centres(self):
        self.assertEqual(
            np.array_equal(
                face_cropper._get_left_and_right_eye_centres([TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0)], [TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0)])[0],
                np.array([0, 1])
            ) and
            np.array_equal(
                face_cropper._get_left_and_right_eye_centres([TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0)], [TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0), TestFaceCropper.Landmark(0, 0)])[1],
                np.array([0, 1])
            ),
            True
        )

        self.assertEqual(
            np.array_equal(
                face_cropper._get_left_and_right_eye_centres([TestFaceCropper.Landmark(0, -1), TestFaceCropper.Landmark(0.5, 0), TestFaceCropper.Landmark(1, 1)], [TestFaceCropper.Landmark(0, -1), TestFaceCropper.Landmark(0.5, 0), TestFaceCropper.Landmark(1, 1)])[0],
                np.array([0.5, 1])
            ) and
            np.array_equal(
                face_cropper._get_left_and_right_eye_centres([TestFaceCropper.Landmark(0, -1), TestFaceCropper.Landmark(0.5, 0), TestFaceCropper.Landmark(1, 1)], [TestFaceCropper.Landmark(0, -1), TestFaceCropper.Landmark(0.5, 0), TestFaceCropper.Landmark(1, 1)])[1],
                np.array([0.5, 1])
            ),
            True
        )


    def test__get_eyes_midpoint(self):
        self.assertEqual(np.array_equal(face_cropper._get_eyes_midpoint([0, 0], [0, 0], (100, 200)), np.array([0, 0])), True)
        self.assertEqual(np.array_equal(face_cropper._get_eyes_midpoint([0.25, 0.25], [0.75, 0.75], (100, 200)), np.array([100, 50])), True)
        self.assertEqual(np.array_equal(face_cropper._get_eyes_midpoint([-0.25, -0.25], [0.75, 0.75], (100, 200)), np.array([50, 25])), True)
        self.assertEqual(np.array_equal(face_cropper._get_eyes_midpoint([0.25, 0.25], [1.25, 1.25], (100, 200)), np.array([150, 75])), True)


    def test__rotate_landmarks(self):
        self.assertEqual(
            np.array_equal(
                face_cropper._rotate_landmarks([TestFaceCropper.Landmark(0.5, 0.5), TestFaceCropper.Landmark(0.5, 0.25)], cv2.getRotationMatrix2D((100, 50), 0, 1), (100, 200)),
                np.column_stack(([100, 50], [100, 25]))
            ),
            True
        )
        self.assertEqual(
            np.array_equal(
                face_cropper._rotate_landmarks([TestFaceCropper.Landmark(0.5, 0.5), TestFaceCropper.Landmark(0.5, 0.25)], cv2.getRotationMatrix2D((100, 50), 90, 1), (100, 200)),
                np.column_stack(([100, 50], [75, 50]))
            ),
            True
        )
        self.assertEqual(
            np.array_equal(
                face_cropper._rotate_landmarks([TestFaceCropper.Landmark(0.5, 0.5), TestFaceCropper.Landmark(0.5, 0.25)], cv2.getRotationMatrix2D((100, 50), -90, 1), (100, 200)),
                np.column_stack(([100, 50], [125, 50]))
            ),
            True
        )
        self.assertEqual(
            np.array_equal(
                face_cropper._rotate_landmarks([TestFaceCropper.Landmark(0.25, 0.5), TestFaceCropper.Landmark(0.5, 0.25)], cv2.getRotationMatrix2D((100, 50), 180, 1), (100, 200)),
                np.column_stack(([150, 50], [100, 75]))
            ),
            True
        )

