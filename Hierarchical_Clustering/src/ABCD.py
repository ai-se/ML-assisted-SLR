from __future__ import division
from pdb import set_trace

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
      Rec = self.TP / (self.TP + self.FN)
      Spec = self.TN / (self.TN + self.FP)
      Prec = self.TP / (self.TP + self.FP)
      Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
      F = 2 * (Prec*Rec) / (Prec+Rec)
      F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
      G = 2 * Rec * Spec / (Rec + Spec)
      G1 = Rec * Spec / (Rec + Spec)
      return [Rec, Spec, Prec, Acc, F, G, self.TP, self.FP, self.TN, self.FN]
    except ZeroDivisionError:
      return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class ABCD():

  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after
    self.result={}

  def __call__(self,type):
    typelist=["Rec", "Spec", "Prec", "Acc", "F", "G", "TP", "FP", "TN", "FN"]
    typeid=typelist.index(type)
    if not self.result:
      uniques = set(self.actual)
      for u in list(uniques):
        self.result[u] = counter(self.actual, self.predicted, indx=u).stats()
    result_tmp={x : self.result[x][typeid] for x in self.result}
    return result_tmp
