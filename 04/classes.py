class Pixel:
    r = 0
    g = 0
    b = 0

    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


class Blob:

    def __init__ (self):
        self.roi = []
        self.pixels_list = []
        self.pixels_qty = 0
