import random
y = random.sample(range(0,9),7)
x = list(map(int, str(int(input()))))
b = 0
for i in range(7):
  if x[i] == y[i]:
    print(x[i],"matches")
    b+=1
  else:
    print(x[i],"does not match") 

print(y)
if b == 7:
    print("You won the lottery")
else:
    print("Screw you. You lose")