a = 0
b = 1
d = 0
while b > 0:
  c = str(input("Enter a word: "))
  if len(c) == 0:
    break
  a += 1
  d += len(c)
print("The average word length is", round(d/a))