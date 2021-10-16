import cv2
from normalised_face_cropper import NormalisedFaceCropper

normalised_face_cropper = NormalisedFaceCropper()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Capture video with default camera (Use DSHOW API for reading to avoid SourceReader warning)

while True:
    read_successful, image = camera.read()
    if not read_successful: raise RuntimeError('Image could not be read from camera!')

    face_image = normalised_face_cropper.crop_face_from_image(image)

    if face_image is None: print("No faces detected!")
    else:
        cv2.imshow('Camera', image)
        cv2.imshow('Normalised face crop', face_image)

    if cv2.pollKey() != -1:  # User pressed key
        camera.release()
        cv2.destroyAllWindows()
        break

