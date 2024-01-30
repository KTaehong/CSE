x = input("Enter color 1: ")
y = input("Enter color 2: ")
if x == 'red':
    if y == 'blue':
        print("purle")
    elif y == 'yellow':
        print("orange")
    else:
        print("Error")
elif x == 'blue':
    if y == 'red':
        print("purle")
    elif y == 'yellow':
        print("green")
    else:
        print("Error")
elif x == 'yellow':
    if y == 'red':
        print("orange")
    elif y == 'blue':
        print("green")
    else:
        print("Error")
else:
    print("Error")