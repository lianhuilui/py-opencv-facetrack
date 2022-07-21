class MPFMDetector:
    def __init__(self):
        import mediapipe as mp

        self.face_loc = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.output_image = None

        return None

    def process_frame(self, input_image):
        import cv2 as cv

        self.output_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(self.output_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                dot_spec = self.mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=0, color=(255, 255, 255))

                self.mp_drawing.draw_landmarks(
                    image=self.output_image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=dot_spec)

        self.output_image = cv.cvtColor(self.output_image, cv.COLOR_RGB2BGR)

        return True


class CVDetector:
    def __init__(self):
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
            cv.rectangle(self.output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return


class MPFDDetector():
    def __init__(self):
        import mediapipe as mp
        import cv2 as cv

        self.face_loc = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        return None

    def process_frame(self, input_image):
        import cv2 as cv
        from mediapipe.framework.formats.location_data_pb2 import LocationData

        rgb_img = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)

        self.output_image = input_image

        if results.detections:
            self.output_image = input_image.copy()

            for detection in results.detections:

                self.nose = self.mp_face_detection.get_key_point(
                    detection, self.mp_face_detection.FaceKeyPoint.NOSE_TIP)

                self.centerpoint = None

                location_data = detection.location_data

                if location_data.format == LocationData.RELATIVE_BOUNDING_BOX:
                    bb = location_data.relative_bounding_box

                    self.centerpoint = (int((bb.xmin + bb.xmin + bb.width)/2 * input_image.shape[1]),
                                        int((bb.ymin + bb.ymin + bb.height)/2 * input_image.shape[0]))

                    self.boundingbox_start = (int(bb.xmin * input_image.shape[1]), int(bb.ymin * input_image.shape[0]))
                    self.boundingbox_end = (int((bb.xmin + bb.width) * input_image.shape[1]), int((bb.ymin + bb.height) * input_image.shape[0]))

                self.mp_drawing.draw_detection(self.output_image, detection)

        return
