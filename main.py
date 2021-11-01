import cv2
from normalised_face_cropper import NormalisedFaceCropper

normalised_face_cropper = NormalisedFaceCropper()
# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Capture video with default camera (Use DSHOW API for reading to avoid SourceReader warning)
camera = cv2.VideoCapture('C:/Users/Toprak/Desktop/Katy Perry - Waking Up In Vegas (Official).mp4')  # Capture video with default camera (Use DSHOW API for reading to avoid SourceReader warning)

while True:
    read_successful, image_bgr = camera.read()
    if not read_successful: raise RuntimeError('Image could not be read from camera!')

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_images_rgb = normalised_face_cropper.get_normalised_faces(image_rgb)

    if face_images_rgb is []: print("No faces detected!")
    else:
        cv2.imshow('Camera', image_bgr)
        for face_id, face_image_rgb in enumerate(face_images_rgb):
            face_image_bgr = cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Face{0}'.format(face_id), face_image_bgr)

    if cv2.pollKey() != -1:  # User pressed key
        camera.release()
        cv2.destroyAllWindows()
        break

