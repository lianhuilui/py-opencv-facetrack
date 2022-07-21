from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self):
        self._centerpoint = None
        self._boundingbox = None

    @abstractmethod
    def process_frame(self, input_image):
        pass

    @property
    def centerpoint(self):
        return self._centerpoint

    @centerpoint.setter
    def centerpoint(self, value):
        self._centerpoint = value

    @property
    def boundingbox(self):
        return self._boundingbox

    @boundingbox.setter
    def boundingbox(self, value):
        self._boundingbox = value

    @property
    def boundingbox_start(self):
        if self.boundingbox and len(self.boundingbox) == 4:
            return (self.boundingbox[0], self.boundingbox[1])
        
        return None

    @property
    def boundingbox_end(self):
        if self.boundingbox and len(self.boundingbox) == 4:
            return (self.boundingbox[2], self.boundingbox[3])
        
        return None

    def parseLocationData(self, location_data, input_image):

        from mediapipe.framework.formats.location_data_pb2 import LocationData

        if location_data.format == LocationData.RELATIVE_BOUNDING_BOX:
            bb = location_data.relative_bounding_box

            self.centerpoint = (int((bb.xmin + bb.xmin + bb.width)/2 * input_image.shape[1]),
                                    int((bb.ymin + bb.ymin + bb.height)/2 * input_image.shape[0]))

            self.boundingbox = [int(bb.xmin * input_image.shape[1]),
                                    int(bb.ymin * input_image.shape[0]),
                                    int((bb.xmin + bb.width) * input_image.shape[1]),
                                    int((bb.ymin + bb.height) * input_image.shape[0])]