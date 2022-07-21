import numpy as np
import cv2 as cv
import utils

face_file = 'env\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'
face_cascade = cv.CascadeClassifier(face_file)

eye_file = 'env\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml'
eye_cascade = cv.CascadeClassifier(eye_file)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

textColor = (255, 255, 255)
lineColor = (50, 50, 50)

pins = ["TOPRIGHT", "TOPLEFT", "BOTTOMRIGHT", "BOTTOMLEFT", "CENTER"]
pin_index = 0

f_counter = 0

face_rects = []
eye_rects = []

fps = 12

face_overlay = None

update = False

use_blank_canvas = False

fit = 240

small_img = np.zeros((fit, fit, 3), np.uint8)

small_img.fill(255)

img = None

cv.namedWindow('fit', cv.WINDOW_AUTOSIZE)
cv.setWindowProperty('fit', cv.WND_PROP_TOPMOST, 1)
cv.setWindowTitle('fit', "FaceTrack")

 # max pixels per frame
max_delta = 1 # 1 means 1 pixel per frame update
min_delta = 5
buffer_delta = (0, 0)

prev_target_center = None
old = None

while True:
    f_counter += 1
    ret, cam_img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    update = fps == 24 or (f_counter % (24//fps) == 0)

    if use_blank_canvas:
        img = np.zeros((cam_img.shape[0], cam_img.shape[1], 3), np.uint8)
    else:
        img = cam_img.copy()

    if update:
        face_overlay = None
        face_rects = []
        eye_rects = []

        # gray = cam_img
        gray = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)

        # faces
        face_cascade
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # take only 1 face
        big_face = utils.biggest_face(faces)
        if big_face is not None:
            faces = [big_face]

        for (x, y, w, h) in faces:
            face_rects.append((x, y, x+w, y+h))

            face_overlay = cam_img[y:y+h, x: x+w]

            # cv.rectangle(cam_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv.rectangle(small_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            eyes = eye_cascade.detectMultiScale(gray[y: y+h, x: x+w], 1.1, 4)

            if len(eyes) >= 2:
                eyes = utils.biggest_eyes(eyes, max=2)

            for (ex, ey, ew, eh) in eyes:
                eye_rects.append((x+ex, y+ey, x+ex+ew, y+ey+eh))
                # cv.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

    # cv.putText(frame, overlay, (0,
    #            frame.shape[0]), cv.FONT_HERSHEY_PLAIN, 1, textColor, 2, cv.LINE_AA)

    if len(face_rects) > 0:
        # print(face_rects[0])
        (x, y, w, h) = face_rects[0]
        w -= x
        h -= y

        # fit = max(fit, w)

        try:
            target_x = x - ((fit-w)//2)
            target_y = y - ((fit-h) // 2)

            target_center = utils.calc_center(target_x, target_y, fit, fit)

            if prev_target_center is not None and old is not None:
                old_x, old_y = old

                diff = utils.diff(target_center, prev_target_center)
                buffer_delta = utils.sub(target_center, prev_target_center)

                if (abs(buffer_delta[0]) > min_delta or abs(buffer_delta[1]) > min_delta):
                    # if target_x < old_x: target_x = max(target_x, old_x - max_delta)
                    # if target_y < old_y: target_y = max(target_y, old_y - max_delta)
                    # if target_x > old_x: target_x = min(target_x, old_x + max_delta)
                    # if target_y > old_y: target_y = min(target_y, old_y + max_delta)

                    clamped = utils.clamp(buffer_delta, max_delta)

                    the_target = utils.add(old, clamped)

                    # utils.overlay_img(img, cam_img, (x - ((fit-w)//2), y - ((fit-h) // 2), fit, fit), pin=pins[pin_index % len(pins)])
                    # print(diff, 'moving...', utils.diff((target_x, target_y), (old_x, old_y)), 'to', (target_x, target_y))

                    # utils.overlay_img(small_img, cam_img, (target_x, target_y, fit, fit), pin=pins[pin_index % len(pins)])
                    # prev_target_center = utils.calc_center(target_x, target_y, fit, fit)
                    # old = (target_x, target_y)
                    # buffer_delta = utils.sub(buffer_delta, (target_x, target_y))

                    utils.overlay_img(small_img, cam_img, (the_target[0], the_target[1], fit, fit), pin=pins[pin_index % len(pins)])
                    prev_target_center = utils.calc_center(the_target[0], the_target[1], fit, fit)
                    old = (the_target[0], the_target[1])
                    buffer_delta = utils.sub(buffer_delta, clamped)

                else:
                    # do nothing
                    # print(diff, 'movement too small. staying still', 'at', (old))
                    utils.overlay_img(small_img, cam_img, (old_x, old_y, fit, fit), pin=pins[pin_index % len(pins)])
                    # utils.overlay_img(small_img, cam_img, (target_x, target_y, fit, fit), pin=pins[pin_index % len(pins)])

            else:
                # utils.overlay_img(img, cam_img, (x - ((fit-w)//2), y - ((fit-h) // 2), fit, fit), pin=pins[pin_index % len(pins)])
                utils.overlay_img(small_img, cam_img, (target_x, target_y, fit, fit), pin=pins[pin_index % len(pins)])
                prev_target_center = target_center
                old = (target_x, target_y)


        except:
            print("exception raised")
    else:
        if old is not None:
            utils.overlay_img(small_img, cam_img, (old[0], old[1], fit, fit), pin=pins[pin_index % len(pins)])



    # cv.setWindowProperty('frame', cv.WND_PROP_OPENGL, cv.WINDOW_OPENGL)
    # cv.namedWindow('cam', cv.WINDOW_KEEPRATIO)

    if len(face_rects) > 0:
        (x, y, w, h) = face_rects[0]
        cv.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

    if len(eye_rects) > 0:
        for eye in eye_rects:
            (x, y, w, h) = eye
            # cv.rectangle(img, (x, y), (w, h), (0, 0, 0), 1)
            cv.circle(img, (x + ((w-x)//2) , y + ((h-y)//2)), 20, (0,0,0), 2, cv.LINE_AA)

    cv.imshow('frame', img)

    cv.imshow('fit', small_img)
    cv.imshow('cam', cam_img)

    key = cv.waitKey(1)

    if key == ord('c'):
        print("c pressed")
        pin_index += 1

    if key == ord('f'):
        print("f pressed")
        use_blank_canvas = not use_blank_canvas

    if key == ord('m'):
        print('m pressed')
        cv.moveWindow('fit', 1672, 760)

    if key == ord('q'):
        print("q pressed")
        break

    print(buffer_delta)
    print('o:',old)

cap.release()
cv.destroyAllWindows()
