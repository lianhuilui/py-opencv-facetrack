import detectors
import tracker
import cv2 as cv

def main():
    mp_fm_detector = detectors.MPFMDetector()
    mp_fd_detector = detectors.MPFDDetector()
    cv_detector = detectors.CVDetector()
    facetracker = tracker.Tracker(mp_fd_detector)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    while True:
        capture_success, cam_img = cap.read()
        if not capture_success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow('main', cam_img)

        cam_img.flags.writeable=False

        # use mediapipe face mesh
        mp_fm_detector.process_frame(cam_img)
        cv.imshow('mp', mp_fm_detector.output_image)

        mp_fd_detector.process_frame(cam_img)
        cv.imshow('mp_fd', mp_fd_detector.output_image)

        # use opencv face detector 
        cv_detector.process_frame(cam_img)
        cv.imshow('cv', cv_detector.output_image)

        # track face
        facetracker.process_frame(cam_img)
        cv.imshow('tracked', facetracker.output_image)

        key = cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()


main()