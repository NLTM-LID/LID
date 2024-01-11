## Python installed libraries
from tkinter import *
import pygame
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import askopenfile
import shutil
import random
import string
import matplotlib.pyplot as plt
from datetime import datetime
## Libraries from external python code
from extract import extract_BNF
from demo_uvector_wssl import *
from sound import *


# Function to perform audio feature extraction and language identification using WSSL uVector
def classification_wssl_uvector():
    # Define supported audio file types
    mask_list = [("Sound files", "*.wav")]

    # To get the BNF features of the selected audio file
    selected_audio = filedialog.askopenfilename(initialdir='', filetypes=mask_list)
    if len(selected_audio) > 0:
        bnf = extract_BNF(selected_audio)

        # Perform language identification using uvector models
        lang, prob_all_lang = uvector_wssl(bnf)

        # Print language identification results
        # print(prob_all_lang)

        # Define language mappings for display
        lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'pun': 9, 'tam': 10, 'tel': 11}
        id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}

        # Get the identified language
        Y1 = id2lang[lang]

        # Display language information using a message box
        # messagebox.showinfo("Given audio language is", Y1)
        answer = messagebox.askyesno(title='Identification Result and Confirmation', message="The predicted language of given audio is {}, \n\nPlease confirm that predicted language is correct?".format(Y1))
        if answer:
            new_filename = datetime.now().strftime("%Y%m%d-%H%M%S")+str(random.randint(1,1000))+'.wav'
            shutil.copy(selected_audio, './classified_audio/{}/{}'.format(Y1, new_filename))
        else:
            new_filename = datetime.now().strftime("%Y%m%d-%H%M%S")+str(random.randint(1,1000))+'.wav'
            shutil.copy(selected_audio, './unclassified_audio/{}'.format(new_filename))

        # Plot the language identification probabilities
        fig = plt.figure(figsize=(10, 5))
        plt.bar(lang2id.keys(), prob_all_lang, color='maroon', width=0.4)
        plt.yscale("log")
        plt.xlabel("Languages")
        plt.ylabel("Language Identification Probability (in log scale)")
        plt.title("Language Identification Probability of Spoken Audio using WSSL uVector")
        plt.show()

# Function to play selected sound file
def play_sound():
    mask_list = [("Sound files", "*.wav")]
    sound_file = filedialog.askopenfilename(initialdir='', filetypes=mask_list)
    if len(sound_file) > 0:
        # Load and play the selected sound file using pygame
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

# Function to stop the currently playing sound
def stop_sound():
    pygame.mixer.music.stop()

# Function to display the main GUI frame with buttons
def GUI_Frame():
    
    # Button to record audio
    record_btn = Button(MainFrame, text="Start Recording", command=lambda m=1:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=10)
    # Button to stop audio recording
    stop_record_btn = Button(MainFrame, text="Stop Recording", command=lambda m=2:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=10)
    # Button to play recorded audio
    play_record_btn = Button(MainFrame, text="Play Recording", command=lambda m=3:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=10)
    
    # Button to play selected audio file
    play_saved_audio = Button(MainFrame, text="Play Saved Audio", command=play_sound, fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=10)
    
    # Button to stop playing audio
    stop_saved_audio = Button(MainFrame, text="Stop Saved Audio", command=stop_sound, fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=10)
    
    # Button to perform audio prediction and language identification using WSSL uVector
    classify2 = Button(MainFrame, text="Identify Language\n(Using WSSL uVector)", fg="white", bg="OrangeRed4", command=classification_wssl_uvector, activeforeground="black", activebackground="coral", relief="raised", bd=12, font = ('calibri', 10, 'bold'))
    
    # Position of above buttons
    record_btn.place(x=25, y=25)
    stop_record_btn.place(x=225, y=25)
    play_record_btn.place(x=425, y=25)
    play_saved_audio.place(x=115, y=100)
    stop_saved_audio.place(x=315, y=100)
    classify2.place(x=150, y=190)


# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Create the main GUI window
MainFrame = Tk()
MainFrame.geometry("600x300")
MainFrame.title("Indian Spoken Language Identification")
MainFrame.configure(background="#D9D8D7")

# Create the menu bar
menubar = Menu(MainFrame)
menubar.add_command(label="", activebackground="OrangeRed4", activeforeground="black", command=GUI_Frame)
MainFrame.config(menu=menubar)

# Create the first frame
GUI_Frame()

# Start the main loop
MainFrame.mainloop()
