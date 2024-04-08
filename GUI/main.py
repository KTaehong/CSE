from tkinter import *

#Create Window
root = Tk()
root.geometry("400x200")

#Add Title
root.title("Avg Calc")

#Add Label Widgets
score1_label = Label(root, text = 'Enter score 1')
score1_label.grid(row=0, column = 0)
score2_label = Label(root, text = 'Enter score 2')
score2_label.grid(row=2, column = 0)
score3_label = Label(root, text = 'Enter score 3')
score3_label.grid(row=4, column = 0)
avg_label = Label(root, text = 'Avg Label: ')
avg_label.grid(row=6, column = 0)

#Add Entry Widgets
e1 = IntVar()
e2 = IntVar()
e3 = IntVar()
score1_entry = Entry(root, textvariable = e1)
score2_entry = Entry(root, textvariable = e2)
score3_entry = Entry(root, textvariable = e3)
score1_entry.grid(row=0, column = 2)
score2_entry.grid(row=2, column = 2)
score3_entry.grid(row=4, column = 2)

def get_average(score1_entry, score2_entry,score3_entry):
    sum = float(score1_entry.get()) + float(score2_entry.get()) + float(score3_entry.get())
    calc_avg = sum/3
    calc_avg_label = Label(root,text = "%.2f"%calc_avg)
    calc_avg_label.grid(row = 6, column = 1)

    pass

#Add Button
avg_button = Button(root,text = "Average", command = lambda:get_average(score1_entry,score2_entry,score3_entry))
avg_button.grid(row = 7, column = 0)
quit_button = Button(root, text = "Quit", command=root.destroy)
quit_button.grid(row=7, column = 2)
#Run the Window
root.mainloop()



