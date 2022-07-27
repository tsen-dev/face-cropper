import unittest
import normalised_face_cropper
import numpy as np


class TestNormalisedFaceCropper(unittest.TestCase):
    def test_get_face_roll_angle(self):
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([1, 0], [0, 0]), 0)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([0, 1], [0, 0]), 90)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([0, 0], [0, 1]), -90)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([0, 0], [1, 0]), 180)

        self.assertEqual(normalised_face_cropper._get_face_roll_angle([1, 1], [0, 0]), 45)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([0, 1], [1, 0]), 135)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([0, 0], [1, 1]), 225)
        self.assertEqual(normalised_face_cropper._get_face_roll_angle([1, 0], [0, 1]), -45)

    def test_crop_within_bounds(self):
        image = np.array([i for i in range(50 * 100)]).reshape((50, 100))

        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, 0, 49, 0, 99), image), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, -50, 200, -50, 200), image), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, 25, 75, 50, 150), image[25:, 50:]), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, 25, 75, -50, 50), image[25:, :50+1]), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, -25, 25, -50, 50), image[:25+1, :50+1]), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._crop_within_bounds(image, -25, 25, 50, 150), image[:25+1, 50:]), True)

    def test_inflate_face_image(self):
        class FaceBox:
            def __init__(self, xmin, width, ymin, height):
                self.xmin = xmin
                self.width = width
                self.ymin = ymin
                self.height = height

        image = np.array([i for i in range(50 * 100)]).reshape((50, 100))

        self.assertEqual(np.array_equal(normalised_face_cropper._inflate_face_image(image, FaceBox(0.50, 0.25, 0.50, 0.25), 0),image[25:38+1, 50:75+1]), True)
        self.assertEqual(np.array_equal(normalised_face_cropper._inflate_face_image(image, FaceBox(0.50, 0.25, 0.50, 0.25), 1), image[19:44+1, 38:88+1]), True)
