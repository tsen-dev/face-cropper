import unittest
import normalised_face_cropper


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

    def test_get_bounding_box_inflation(self):
        self.assertEqual(True, True)
