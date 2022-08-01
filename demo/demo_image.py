import cv2
from face_cropper import FaceCropper

face_cropper = FaceCropper()

image_bgr = cv2.imread('demo_1.jpg')
if image_bgr is None: raise RuntimeError('Image could not be read')

faces_rgb = face_cropper.get_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

if not faces_rgb:
    print("No faces detected")
else:
    cv2.imshow('Image', image_bgr)
    for face_id, face_rgb in enumerate(faces_rgb):
        cv2.imshow('Face {0}'.format(face_id), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()
