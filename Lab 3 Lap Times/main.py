a = int(input("Enter number of laps: "))
b = 0
e = 100000000
d = 0
for i in range(a):
  c = float(input("Enter time for Lap " + str(i+1) + ": "))
  if c > b:
    b = c
  if c < e:
    e = c
  d+=c
print("Fastest lap time:", e)
print("Slowest lap time:", b)
print("Average lap time:", d/a)