def determine_grade(x):
  if x in range(90, 101):
    print("A")
  elif x in range(80, 90):
    print("B")
  elif x in range(70, 80):
    print("C")
  elif x in range(60, 70):
    print("D")
  elif x in range(0, 60):
    print("F")

def calc_average(a,b,c,d,e):
  return (a+b+c+d+e)/5


a=int(input("Input test score: "))
determine_grade(a)
b=int(input("Input test score: "))
determine_grade(b)
c=int(input("Input test score: "))
determine_grade(c)
d=int(input("Input test score: "))
determine_grade(d)
e=int(input("Input test score: "))
determine_grade(e)
determine_grade(calc_average(a,b,c,d,e))
