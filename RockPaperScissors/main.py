import random
random = random.randint(1,3)
choice = str(input("Enter rock, paper, or scissors: "))
def choose(x,y):
  if y.lower() == 'rock':
    if x == 1:
      print("You Tied")
    elif x == 3:
      print("You Won")
    elif x == 2:
      print("You Lost")
  elif y.lower() == 'paper':
    if x == 2:
      print("You Tied")
    elif x == 1:
      print("You Won")
    elif x == 3:
      print("You Lost")
  elif y.lower() == 'scissors':
    if x == 3:
      print("You Tied")
    elif x == 2:
      print("You Won")
    elif x == 1:
      print("You Lost")