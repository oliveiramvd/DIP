class Pixel:
    r = 0
    g = 0
    b = 0

    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


class Blob:

    def __init__ (self, img_width, img_height):
        self.min_x = img_width-1
        self.min_y = img_height-1
        self.max_x = 0
        self.max_y = 0
        self.pixels_list = []
        self.pixels_qty = 0
