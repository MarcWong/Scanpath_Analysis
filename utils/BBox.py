import numpy as np

#################################################################
class BBox:
    def __init__(self, name: str = '', coords: list = [], text: str = '', color: str = '', id: int = -1):
        self.name = name
        self.coords = np.array(coords)
        self.text = text
        self.color = color
        self.id = get_bboxid_by_name(name)
        # print('init bbox', self.name, self.coords, self.text, self.id)

# Bbox index for semantic regions of annotations:
def get_bboxid_by_name(name):
    id1 = -1
    if 'a_' in name:
        id1 = 1
    elif 'b_' in name:
        id1 = 2
    elif 'c_' in name:
        id1 = 3
    elif 'title' in name:
        id1 = 4
    elif 'legend' in name:
        id1 = 5
    elif 'axis' in name:
        id1 = 6
    elif 'mark' in name:
        id1 = 7
    else:
        print('Error: Unknown name: %s' % name)
    return id1

