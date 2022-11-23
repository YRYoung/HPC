import time

n = 100

time1 = time.time()
"""获取计算前的系统时间"""

w = n + 10  # 多计算10位，防止尾数取舍的影响
b = 10 ** w  # 算到小数点后w位
x1 = b * 4 // 5  # 求含4/5的首项
x2 = b // -239  # 求含1/239的首项
he = x1 + x2  # 求第一大项
n *= 2  # 设置下面循环的终点，即共计算n项

for i in range(3, n, 2):
    x1 //= -25  # 求每个含1/5的项及符号
    x2 //= -57121  # 求每个含1/239的项及符号
    x = (x1 + x2) // i  # 求两项之和
    he += x  # 求总和
    pai = he * 4  # 求π的值
    pai //= 10 ** 10  # 舍掉后十位

file_name = 'pi_million_digits.txt'
with open(file_name, 'w') as file_object:
    file_object.write(str(pai))

time2 = time.time()
print("总耗时：" + str(time2 - time1) + "s")
