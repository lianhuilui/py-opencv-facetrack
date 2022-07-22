from detectors.base import BaseDetector

class MPFDDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        import mediapipe as mp
        import cv2 as cv

        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_face_detection = mp.solutions.face_detection

        self._face_detection = self._mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        return None


    def process_frame(self, input_image):
        import cv2 as cv
        from mediapipe.framework.formats.location_data_pb2 import LocationData

        rgb_img = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb_img)

        self.output_image = input_image

        if results.detections:
            self.output_image = input_image.copy()

            for detection in results.detections:

                self.nose = self._mp_face_detection.get_key_point(
                    detection, self._mp_face_detection.FaceKeyPoint.NOSE_TIP)

                self.parseLocationData(detection.location_data, input_image)

                self._mp_drawing.draw_detection(self.output_image, detection)
