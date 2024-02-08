def maximum(x,y):
  if x > y:
    return x
  elif y > x:
    return y
  else: 
    return "none of them"
a = int(input("Enter the first number: "))
b = int(input("Enter the second number: "))
print("The greater value is", maximum(a,b))