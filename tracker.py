from detectors.base import BaseDetector


class Tracker():
    def __init__(self, detector: BaseDetector):
        self.detector = detector
        pass


    def set_detector(self, value):
        self.detector = value

    def process_frame(self, image):
        import cv2 as cv

        self.output_image = image.copy()

        if self.detector.centerpoint:
            cv.circle(self.output_image, self.detector.centerpoint,
                      1, (255, 255, 255), 1, cv.LINE_AA)

        if self.detector.boundingbox:
            cv.rectangle(self.output_image, self.detector.boundingbox_start, self.detector.boundingbox_end,
                         (255, 0, 0), 2, cv.LINE_AA)
