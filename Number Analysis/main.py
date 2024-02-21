a = []
b = 0
for x in range(20):
    a.append(int(input()))
    b += a[x]
    if x == 19:
        print("Max:", max(a))
        print("Min:", min(a))
        print("Number of elements: ", 20)
        print("Total:", b / 20)
