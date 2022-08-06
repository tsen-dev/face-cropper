import cv2
from face_cropper import FaceCropper

face_cropper = FaceCropper()

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Capture video with default camera (Use DSHOW API for reading to avoid SourceReader warning)
# camera = cv2.VideoCapture('demo_1.mp4')  # Read video from specified path

while True:
    read_successful, image_bgr = camera.read()
    if not read_successful: raise RuntimeError('Image could not be read!')

    faces_rgb = face_cropper.get_faces_debug(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not faces_rgb:
        print("No faces detected!")
    else:
        cv2.imshow('Image', image_bgr)
        for face_id, face_rgb in enumerate(faces_rgb):
            cv2.imshow('Face {0}'.format(face_id), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

    if cv2.pollKey() != -1:  # User pressed key
        camera.release()
        cv2.destroyAllWindows()
        break

