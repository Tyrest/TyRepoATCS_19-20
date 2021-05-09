import random

sum = 0
for i in range(0, 1000000):
    left = 10.0
    cars = 0
    while left >= 1:
        left -= random.uniform(1, 2)
        cars += 1
    sum += cars
    left = 10
print(sum / 1000000)
