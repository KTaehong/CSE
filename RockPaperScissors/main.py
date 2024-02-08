import random
#1 rock
#2 paper
#3 Scissor

def choose():
  t=1
  while t == 1:
    y = str(input("Enter rock, paper, or scissors: "))
    x = random.randint(1,3)
    if t == 1:
      if y.lower() == 'rock':
        if x == 1:
          print("You Tied")
        elif x == 3:
          print("You Won")
          t+=1
        elif x == 2:
          print("You Lost")
          t+=1
      elif y.lower() == 'paper':
        if x == 2:
          print("You Tied")
        elif x == 1:
          print("You Won")
          t+=1
        elif x == 3:
          print("You Lost")
          t+=1
      elif y.lower() == 'scissors' or y.lower() == 'scissor':
        if x == 3:
          print("You Tied")
        elif x == 2:
          print("You Won")
          t+=1
        elif x == 1:
          print("You Lost")
          t+=1
      else:
        print("Invalid")
choose()
