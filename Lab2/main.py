x = input("Enter color 1: ")
y = input("Enter color 2: ")
if x.lower() == 'red':
    if y.lower() == 'blue':
        print("purle")
    elif y.lower() == 'yellow':
        print("orange")
    else:
        print("Error")
elif x.lower() == 'blue':
    if y.lower() == 'red':
        print("purle")
    elif y.lower() == 'yellow':
        print("green")
    else:
        print("Error")
elif x.lower() == 'yellow':
    if y.lower() == 'red':
        print("orange")
    elif y.lower() == 'blue':
        print("green")
    else:
        print("Error")
else:
    print("Error")
