import tkinter
import customtkinter
import os
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk()
app.geometry("700x450")

def run_linguei():
    os.system('python3 testing.py')

def create_btn_back():
    button2 = customtkinter.CTkButton(master=app,
                                      text="ATRAS",
                                      fg_color="white",
                                      hover_color="orange",
                                      text_color="black",
                                      command=index)
    button2.place(relx=0.11, rely=.1, anchor=tkinter.CENTER)


def instrucciones():
    image1 = Image.open("instrucciones.png").resize((700, 450))
    test = ImageTk.PhotoImage(image1)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
    create_btn_back()

def index():

    image1 = Image.open("bg.png").resize((700, 450))
    test = ImageTk.PhotoImage(image1)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)


    button = customtkinter.CTkButton(master=app,
                                     text="COMENZAR",
                                     fg_color= "white",
                                     hover_color= "orange",
                                     text_color= "black",
                                     command=run_linguei)
    button.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

    button2 = customtkinter.CTkButton(master=app,
                                     text="INSTRUCCIONES",
                                     fg_color= "white",
                                     hover_color= "orange",
                                     text_color= "black",
                                     command=instrucciones)
    button2.place(relx=0.5, rely=.9, anchor=tkinter.CENTER)


    app.mainloop()

def main():
    index()
main()