import utils

def test_max():
    faces = []

    max_face = {'x': 0, 'y':0, 'w': 20, 'h':200}

    faces.append({'x': 0, 'y':0, 'w': 10, 'h':10})
    faces.append({'x': 0, 'y':0, 'w': 10, 'h':100})
    faces.append(max_face)
    faces.append({'x': 0, 'y':0, 'w': 10, 'h':10})
    faces.append({'x': 0, 'y':0, 'w': 100, 'h':10})
    faces.append({'x': 0, 'y':0, 'w': 20, 'h':100})

    assert(utils.biggest_face(faces) == max_face)

if __name__ == "__main__":
    test_max()
    print("Everything passed")