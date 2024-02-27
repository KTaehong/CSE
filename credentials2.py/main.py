# from flask import Flask

# app = Flask('app')
# @app.route('/')
# def hello_world():
#  return 'Hello, World!'
# app.run(host='0.0.0.0', port=8080)


def create_username():
    while True:
        u = input("Enter username: ")
        if "@" in u and "." in u:
            return u
        else:
            print("Invalid")


def create_password():
    while True:
        p = input("Enter password: ")

        if len(p) < 8:
            print("Password must be at least 8 characters long.")
        elif not any(char.isupper() for char in p):
            print("Include an uppercase letter.")
        elif not any(char.islower() for char in p):
            print("Include a lowercase letter.")
        elif not any(char.isdigit() for char in p):
            print("Include a number.")
        else:
            return p


def password_exists(p):
    with open("passwords.txt", "r") as file:
        for line in file:
            if f"{p}" in line:
                return True
    return False


def save_password(p):
    with open("passwords.txt", "a") as file:
        file.write(f"{p}\n")

    print("Password saved.")


print("Create a username and password")
u = create_username()
f = open("passwords.txt", "w")
f.close()
while True:
    p = create_password()
    if not password_exists(p):
        break
    print("Password already exists. Pick another one")

save_password(p)

