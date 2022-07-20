def biggest_face(faces):
    if len(faces) > 0:
        return max(faces, key=lambda x: x[2] * x[3])

    return None


def biggest_eyes(_eyes, max=2):
    eyes = _eyes[:]

    if len(eyes) == max:
        return eyes

    if len(eyes) > max:
        eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)
        return eyes[0:max]

    return eyes

def overlay_img(img, overlay, coords, pin="CENTER"):
    (x, y, w, h) = coords

    x = max(0, x)
    y = max(0, y)

    x = min(x, overlay.shape[1] - w)
    y = min(y, overlay.shape[0] - h)

    if pin == "CENTER":
        img[(img.shape[0] // 2) - (h//2): (img.shape[0] // 2) + h - (h // 2), (img.shape[1] // 2) - (w//2): (img.shape[1] // 2) + w - (w // 2)] = overlay[y:y+h, x:x+w]

    elif pin == "TOPRIGHT":
        img[0:h, img.shape[1]-w:img.shape[1]] = overlay[y:y+h, x:x+w]
    
    elif pin == "BOTTOMLEFT":
        img[img.shape[0]-h:img.shape[0], 0:w] = overlay[y:y+h, x:x+w]

    elif pin == "TOPLEFT":
        img[0:h, 0:w] = overlay[y:y+h, x:x+w]

    else:
        img[img.shape[0]-h:img.shape[0], img.shape[1]-w:img.shape[1]] = overlay[y:y+h, x:x+w]


def calc_center(x, y, w, h):
    return (x + w//2, y + h//2)

def diff(one, two):
    return (abs(one[0] - two[0]), abs(one[1] - two[1]))

def add(one, two):
    return (one[0] + two[0], one[1] + two[1])

def sub(one, two):
    return (one[0] - two[0], one[1] - two[1])

def clamp(delta, max_change):
    x = delta[0]
    y = delta[1]

    if delta[0] < 0 and -delta[0] > max_change:
        x =- max_change
    
    if delta[1] < 0 and -delta[1] > max_change:
        y =- max_change

    if delta[0] > 0 and delta[0] > max_change:
        x = max_change
    
    if delta[1] > 0 and delta[1] > max_change:
        y = max_change

    return (x, y)