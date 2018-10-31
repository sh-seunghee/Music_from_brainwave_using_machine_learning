'''

Author: Seunghee Lee
Last modified: 10/31/2018

In this script, we use python library TKinter to provide a graphical user interface

The module contains following components:

    1. Input recorded brainwave file
    2. Listening to the reconstructed melody from the brainwave file
    3. User select music genre that they want to convert to
    4. Listening to a piece of melody with the user defined genre.

'''

from tkinter import *
from tkinter.ttk import Frame, Button, Style
from tkinter.filedialog import askopenfilename

import os
from utils import *

from brainwave import brainwave_to_melody

class MusicAPP(Frame):

    def __init__(self):

        super().__init__()

        self.initUI()

    def initUI(self):

        self.master.title("Brainwave to Music Conversion Software")
        self.style = Style()

        self.style.theme_use("default")

        self.pack(fill=BOTH, expand=True)

        # Frame 1 ----------------------------------------------

        frame1 = Frame(self, relief=RAISED, borderwidth=1)
        frame1.pack(fill=BOTH, expand=True)

        lbl1 = Label(frame1, text="Press Browse button, "
                                  "then press the submit button", font=(None, 15), background='#d9d9d9')
        lbl1.pack(side=TOP, padx=5, pady=15)

        self.pathlabel = Label(frame1, width=50)
        self.pathlabel.pack(side=LEFT, padx=10, expand=True)

        button_browse = Button(frame1, text="Browse", command=self.openFile)
        button_browse.pack(fill=X, padx=10, expand=True)

        # Parameter selection (channal, samplerate)-----------

        scrollbar1 = Scrollbar(frame1)
        scrollbar1.pack(side=RIGHT, fill=Y)

        self.listbox_channal = Listbox(frame1, selectmode='extended', height=1)

        # attach listbox to scrollbar
        self.listbox_channal.config(yscrollcommand=scrollbar1.set)
        scrollbar1.config(command=self.listbox_channal.yview)

        for i in range(1, 39):
            self.listbox_channal.insert(i, str(i))

        self.listbox_channal.pack(fill=X)

        self.sampleRate_scaler = Scale(frame1, from_=1, to=100, orient=HORIZONTAL)
        self.sampleRate_scaler.set(1)
        self.sampleRate_scaler.pack()

        #-------------------------------------------------------

        closeButton = Button(self, text="Close", command=sys.exit)
        closeButton.pack(side=RIGHT)

        submitButton = Button(self, text="Submit", command=self.melodyFromBrainwave)
        submitButton.pack(side=RIGHT)

    def openFile(self):

        self.filePath = askopenfilename(filetypes=(("All Files", "*.*"), ("Text File", "*.txt")),
                                   title="Choose a brainwave file")
        print(self.filePath)
        self.pathlabel.config(text=self.filePath)

        '''
        try:
            with open(filename, 'r') as inputFile:
                print(inputFile.read())
        except:
            print("No file exists")
        '''

    #def paramterSelectionUI(self):


    def melodyFromBrainwave(self):

        frame2 = Frame(self, relief=RAISED, borderwidth=1)
        frame2.pack(fill=Y, expand=True)

        root.geometry("600x400+200+200")

        # Call brainwave.py module with params to create a melody(in midi format) from brainwave file
        nChannal = self.listbox_channal.curselection()[0]
        sampleRate = self.sampleRate_scaler.get()

        midiFile = brainwave_to_melody(_filename=self.filePath, _nChannal=nChannal, _sampleRate=sampleRate)

        workingDirPath = os.getcwd()
        midiFilePath = os.path.join(workingDirPath, midiFile)

        # Waiting for the midi file creation
        while True:
            exists = os.path.exists(midiFilePath)
            if exists: break

        # Play the generated midi file
        playMusic(midiFilePath)


if __name__ == "__main__":

    root = Tk()
    root.geometry("1000x600+200+100")

    app = MusicAPP()
    root.mainloop()

