import csv


class SummaryWriter(object):
    def __init__(self, nbatches: int, logdir: str, title_prefix=""):
        self.logdir = logdir
        self.nbatches = nbatches
        self.title_prefix = title_prefix
        self.scalar_map = {}

    def add_scalars(self, title: str, value: int, step: int):
        title = self.title_prefix + " " + title
        if title in self.scalar_map:
            self.scalar_map[title][step] = value
        else:
            newlist = [None] * self.nbatches
            newlist[step] = value
            self.scalar_map[title] = newlist

    def to_csv(self):
        header = ["epoch"]
        header.extend(self.scalar_map.keys())

        with open(self.logdir, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for x in range(0, self.nbatches):
                entry = [x]
                for k in self.scalar_map:
                    entry.append(self.scalar_map[k][x])
                writer.writerow(entry)
