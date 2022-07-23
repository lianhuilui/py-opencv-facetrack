from detectors.base import BaseDetector


class MPFMDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        import mediapipe as mp

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
        import numpy
        from mediapipe.python.solutions import face_mesh as mp_faces

        self.output_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(self.output_image)

        cal_avg = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = []
                ys = []

                for idx, landmark in enumerate(face_landmarks.landmark):
                    xs.append(landmark.x)
                    ys.append(landmark.y)

                if cal_avg:

                    self.centerpoint = (
                        int(numpy.average(xs) * input_image.shape[1]),
                        int(numpy.average(ys) * input_image.shape[0])
                    )

                    w = int((max(xs) - min(xs)) * input_image.shape[1])
                    h = int((max(ys) - min(ys)) * input_image.shape[0])

                    self.boundingbox = [
                        self.centerpoint[0] - w//2,
                        self.centerpoint[1] - h//2,
                        self.centerpoint[0] + w//2,
                        self.centerpoint[1] + h//2
                    ]
                else:
                    self.boundingbox = [int(min(xs) * input_image.shape[1]),
                                        int(min(ys) * input_image.shape[0]),
                                        int(max(xs) * input_image.shape[1]),
                                        int(max(ys) * input_image.shape[0])]

                    self.centerpoint = (
                        (self.boundingbox[0] + self.boundingbox[2]) // 2,
                        (self.boundingbox[1] + self.boundingbox[3]) // 2)

                dot_spec = self.mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1, color=(255, 255, 255))

                line_spec = self.mp_drawing.DrawingSpec(
                    thickness=1, color=(255, 255, 255)
                )

                self.mp_drawing.draw_landmarks(
                    self.output_image,
                    face_landmarks,
                    mp_faces.FACEMESH_TESSELATION,
                    landmark_drawing_spec=dot_spec,
                    connection_drawing_spec=line_spec
                )

        self.output_image = cv.cvtColor(self.output_image, cv.COLOR_RGB2BGR)

        return True
