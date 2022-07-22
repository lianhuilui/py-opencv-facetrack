from detectors.base import BaseDetector


class Tracker():
    def __init__(self, detector: BaseDetector = None, crop_dim = None):
        self.detector = detector
        self.crop_dim = crop_dim
        self._annotate = True
        pass


    def toggle_annotate(self):
        self._annotate = not self._annotate

    def set_detector(self, value):
        self.detector = value

    def process_frame(self, image):
        import cv2 as cv

        self.output_image = image.copy()

        if self._annotate:
            if self.detector.centerpoint:
                cv.circle(self.output_image, self.detector.centerpoint,
                        1, (255, 255, 255), 1, cv.LINE_AA)

            if self.detector.boundingbox:
                cv.rectangle(self.output_image, self.detector.boundingbox_start, self.detector.boundingbox_end,
                            (255, 0, 0), 2, cv.LINE_AA)

        if self.crop_dim and self.detector.centerpoint:
            w, h = self.crop_dim
            x = self.detector.centerpoint[0] - (w // 2)
            y = self.detector.centerpoint[1] - (h // 2)

            x = max (0, x)
            y = max (0, y)
            x = min (self.output_image.shape[1] - w, x)
            y = min (self.output_image.shape[0] - h, y)

            self.output_image = self.output_image[y:y+h, x:x+w]
