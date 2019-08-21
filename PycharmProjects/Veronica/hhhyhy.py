import math
import random
import statistics

class Orange:
    def __init__(self,w,c):
        self.weiths = w
        self.color = c
        self.mold = 0
        print("Create")

    def rot(self, day,temp):
        self.mold = day * temp

org1 = Orange(100, "dark orange")
print (org1.weiths)
print (org1.color)
org2 = Orange(400, "dark orange")
print (org2.weiths)
print (org2.color)
org3 = Orange(300, "ligth orange")
print (org3.weiths)
print (org3.color)