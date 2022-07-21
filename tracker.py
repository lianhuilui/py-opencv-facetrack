class Tracker():
    def __init__(self, detector):
        self.detector = detector
        pass

    def process_frame(self, image):
        import cv2 as cv

        self.output_image = image.copy()

        if self.detector.centerpoint and self.detector.boundingbox_start and self.detector.boundingbox_end:
            cv.rectangle(self.output_image, self.detector.boundingbox_start, self.detector.boundingbox_end,
                            (255,0,0), 2, cv.LINE_AA)

            # cv.putText(self.output_image, text, (
            #     0, self.output_image.shape[0]), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv.LINE_AA)

            cv.circle(self.output_image, self.detector.centerpoint, 1, (255,255,255), 1, cv.LINE_AA)
