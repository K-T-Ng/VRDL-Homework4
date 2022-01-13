class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, avg, n):
        self.val += (avg * n)
        self.count += n

    def item(self):
        return self.val / self.count if self.count else 0

if __name__ == '__main__':
    ctr = AvgMeter()
    print(ctr.item())
    ctr.update(10, 1)
    ctr.update(2, 5)
    print(ctr.item())
