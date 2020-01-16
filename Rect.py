class Rect:
    """
    Class for defining a bounding box and apply operations on it.
    """
    def __init__(self, x1, y1, x2, y2, prob=0.0, text=None):
        self.set_coordinates(x1, y1, x2, y2)
        self.prob = prob
        self.text = text

    def set_x2(self, x2):
    	self.x2 = x2
    	self.w = self.x2 - self.x1

    def set_y2(self, y2):
    	self.y2 = y2
    	self.h = self.y2 - self.y1

    def set_coordinates(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.w, self.h = w, h

    def area_diff(self, other):
        return self.area() - self.intersection(other).area()

    def area(self):
        return self.w * self.h

    def intersection(self, other):
        a, b = self, other
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return Rect(x1, y1, x2 - x1, y2 - y1)
        else:
            return type(self)(0, 0, 0, 0)

    __and__ = intersection

    def union(self, other):
        a, b = self, other
        x1 = min(a.x1, b.x1)
        y1 = min(a.y1, b.y1)
        x2 = max(a.x2, b.x2)
        y2 = max(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2 - x1, y2 - y1)

    def contains(p1):
    	if p1[0] < self.x2 and p1[0] > self.x1:
    		if p1[1] < self.y2 and p1[1] > self.y1:
    			return True
    	return False

    __or__ = union
    __sub__ = area_diff

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rect) and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))