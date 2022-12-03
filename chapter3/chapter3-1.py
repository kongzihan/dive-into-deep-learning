"""
本程序包含了《动手学习深度学习》中，第三章 深度学习基础中的部分代码
代码为pytorch版本

"""
import numpy as np
from time import time

# =======================     练习 - 1      ========================
# 用numpy实现两个向量相加
# 并比较相加两个向量的两种方法的运行时间

vec1 = np.ones(shape=10000)
vec2 = np.ones(shape=10000)

start1 = time()
sum1 = np.ones(shape=10000)
for i in range(10000):
    sum1[i] = vec1[i] + vec2[i]
time1 = time() - start1
print("标量加法运算的运行时间为：", time1)

start2 = time()
sum2 = vec1 + vec2
time2 = time() - start2
print("矢量加法运算的运行时间为：", time2)


