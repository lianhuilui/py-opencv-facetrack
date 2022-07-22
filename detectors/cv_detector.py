from detectors.base import BaseDetector


class CVDetector(BaseDetector):
    def __init__(self):
        super().__init__()

        import cv2 as cv

        face_file = 'env\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'
        self.face_cascade = cv.CascadeClassifier(face_file)

        return

    def process_frame(self, input_image):
        import cv2 as cv

        gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        self.output_image = input_image.copy()

        for (x, y, w, h) in faces:
            self.centerpoint = (int((x + x + w)/2), int((y + y + h)/2))
            self.boundingbox = [x, y, x+w, y+h]

            cv.rectangle(self.output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
