import detectors
from detectors.cv_detector import CVDetector
from detectors.mp_fm_detector import MPFMDetector 
import tracker
import cv2 as cv

TOTAL_CAP_DEVICES = 1 # set this to more than 1 to process multiple cams at once
TRACKER_RES = (220,220)

# Key Bindings
HOTKEY_TOGGLE_ANNOTATE_TRACKER = 'a'
HOTKEY_CYCLE_FACE_DETECTION_METHOD = 't'

def main():
    current_detector = 0

    detector_list = []
    trackers_list = []

    caps = []
    for i in range(TOTAL_CAP_DEVICES):
        # TODO: allow setting camera feed's resolution
        cap = cv.VideoCapture(i)

        if cap.isOpened():
            caps.append(cap)

            mp_fm_detector = detectors.MPFMDetector()
            mp_fd_detector = detectors.MPFDDetector()
            cv_detector = detectors.CVDetector()
            detector_list.append([mp_fm_detector, mp_fd_detector, cv_detector])
            trackers_list.append(tracker.Tracker(crop_dim=TRACKER_RES))

    if not caps:
        print("no capture devices present or available")
        return None

    while True:
        cam_imgs = []

        for cap in caps:
            capture_success, cam_img = cap.read()
            if capture_success:
                cam_imgs.append(cam_img)

        if cam_imgs:
            for idx, cam_img in enumerate(cam_imgs):
                d = detector_list[idx][current_detector % len(detector_list[idx])]
                facetracker = trackers_list[idx]

                facetracker.set_detector(d)

                cam_img.flags.writeable=False

                # detect face
                d.process_frame(cam_img)
                cv.imshow('CAM %d face detection' % idx, d.output_image)

                # track face
                facetracker.process_frame(cam_img)
                cv.imshow('CAM %d tracked' % idx, facetracker.output_image)

                key = cv.waitKey(1)

                if key == ord(HOTKEY_TOGGLE_ANNOTATE_TRACKER):
                    for t in trackers_list:
                        t.toggle_annotate()

                if key == ord(HOTKEY_CYCLE_FACE_DETECTION_METHOD):
                    current_detector += 1
        else:
            print ("could not get frame from at least one device")
            break

    cap.release()
    cv.destroyAllWindows()


main()