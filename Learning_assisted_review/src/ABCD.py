from __future__ import division


class counter():
    def __init__(self, before, after, indx):
        self.indx = indx
        self.actual = before
        self.predicted = after
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        for a, b in zip(self.actual, self.predicted):
            if a == indx and b == indx:
                self.TP += 1
            elif a == b and a != indx:
                self.TN += 1
            elif a != indx and b == indx:
                self.FP += 1
            elif a == indx and b != indx:
                self.FN += 1
            elif a != indx and b != indx:
                pass

    def stats(self):
        try:

            Sen = self.TP / (self.TP + self.FN)
            Spec = self.TN / (self.TN + self.FP)
            Prec = self.TP / (self.TP + self.FP)
            Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
            F = 2 * (Prec * Sen) / (Prec + Sen)
            F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
            G = 2 * Sen * Spec / (Sen + Spec)
            G1 = Sen * Spec / (Sen + Spec)
            return {
                "Sen" : Sen,
                "Prec": Prec,
                "Spec": Spec,
                "Acc" : Acc,
                "F1"  : F,
                "G1"  : G
            }

        except ZeroDivisionError:
            return {
                "Sen" : 0,
                "Prec": 0,
                "Spec": 0,
                "Acc" : 0,
                "F1"  : 0,
                "G1"  : 0
            }

class ABCD():
    "Statistics Stuff, confusion matrix, all that jazz..."

    def __init__(self, before, after):
        self.actual = before
        self.predicted = after

    def __call__(self):
        uniques = set(self.actual)
        for u in list(uniques):
            yield counter(self.actual, self.predicted, indx=u)
