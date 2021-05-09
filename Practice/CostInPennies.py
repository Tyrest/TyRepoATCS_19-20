cost = 0
good = False
while not good:
    foo = input("What is your cost in pennies: ")
    try:
        cost = int(foo)
        break
    except ValueError as verror:
        print("Not in cents")
    try:
        cost = int(float(foo) * 100)
        break
    except ValueError as verror:
        print("Not valid number")
print(cost)
