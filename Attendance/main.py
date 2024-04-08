import pickle
from consolemenu import *
from consolemenu.items import *

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
stat = ["p", "a", "t"]

try:
    with open("attendance.pkl", "rb") as file:
        attendanceFile = pickle.load(file)
except:
    attendanceFile = {}


# Good

def addStudent():
    print("Input q to quit")
    while True:
        name = input("Enter student name: ")
        if name.lower() == "q":
            break
        elif name not in attendanceFile:
            attendanceFile[name] = {x: "Unavailable" for x in days}
        else:
            print("Name already exists")
    with open("attendance.pkl", "wb") as file:
        pickle.dump(attendanceFile, file)


def takeAttendance():
    print("1. Monday\n2. Tuesday\n3. Wednesday\n4. Thursday\n5. Friday")
    x = int(input("Enter the corresponding number: "))
    sday = days[x - 1]
    print(f"Attendance: {sday}. Options are P(resent), A(bsent), and T(ardy):")
    for name, a in attendanceFile.items():
        while True:
            status = input(f"{name}: ").lower()
            if status in stat:
                a[sday] = status
                break
        else:
            print("Invalid. Please input P, A, or T")
    with open("attendance.pkl", "wb") as file:
        pickle.dump(attendanceFile, file)


# Edit this one
def editAttendance():
    while True:
        name = input("Enter student's name: ")
        if name in attendanceFile:
            print(f"Attendance for {name} (P for Present, A for Absent, T for Tardy):")
            for day, status in attendanceFile[name].items():
                print(f"{day}: {status.upper()}")

            while True:
                choice = input("Enter the corresponding number to edit the day or 'q' to quit: ")
                if choice.lower() == 'q':
                    break
                elif choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(days):
                        eday = days[choice - 1]
                        new = input("Enter new attendance (P(resent), A(bsent), and T(ardy)): ").lower()
                        if new in stat:
                            attendanceFile[name][eday] = new
                            break
                        else:
                            print("Invalid input. Please enter P, A, or T.")
                    else:
                        print("Invalid choice. Please enter a number between 1 and 5 or 'q' to quit.")
                else:
                    print("Invalid input. Please enter a number between 1 and 5 or 'q' to quit.")
            break
        else:
            print("Student not found.")

    with open("attendance.pkl", "wb") as file:
        pickle.dump(attendanceFile, file)


def printStudentAttendance():
    name = input("Enter student's name: ")
    if name in attendanceFile:
        print(f"Attendance for {name} (P(resent), A(bsent), and T(ardy)):")
        for day, status in attendanceFile[name].items():
            print(f"{day}: {status}")
    else:
        print("Student not found.")
    return input("Press Enter to quit")


def printClassAttendance():
    print("Class Attendance: (P(resent), A(bsent), and T(ardy)):")
    for student, attendance in attendanceFile.items():
        print(student)
        for day, status in attendance.items():
            print(f"{day}: {status.upper()}")
        print()  # Add an empty line after each student's attendance
    return input("Press Enter to quit")


menu = ConsoleMenu("Class Attendance", "by ttangkong")

attendance = FunctionItem("Take Attendance", takeAttendance)

edit = FunctionItem("Edit Attendance", editAttendance)

sPrint = FunctionItem("Print Student Attendance", printStudentAttendance)

cPrint = FunctionItem("Print Class Attendance", printClassAttendance)

add = FunctionItem("Add a Student", addStudent)

menu.append_item(attendance)
menu.append_item(edit)
menu.append_item(sPrint)
menu.append_item(cPrint)
menu.append_item(add)

menu.show()

