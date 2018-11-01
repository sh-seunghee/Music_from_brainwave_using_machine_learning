'''
Author: Seunghee Lee
Last modified: 11/01/2018

In this script, we use python library TKinter to provide a graphical user interface.

The module contains following components:

    1. Input recorded brainwave file
    2. Get channel and rate values from user
    3. Play a reconstructed melody from the brainwave file
    4. User select music genre that they want to convert to
    5. Play to a piece of melody with the user defined genre.

    To do;
    - connect with genre conversion module
    - Stop listening function should be added
    - add the generated music sheet into the right frame

'''

from tkinter import *
from tkinter.filedialog import askopenfilename

import os
from musicPlayer import *

from brainwave import brainwave_to_melody

LABELBG = '#b5b5b5'

class MusicAPP():

    def __init__(self, parent):

        super().__init__()

        self.parent = parent
        self.musicPlayer = MusicPlayer()
        self.initUI()

    def initUI(self):

        self.parent.title("Brainwave to Music Conversion Software")

        # Controller Frame ----------------------------------------------
        self.controllerFrame = Frame(self.parent, borderwidth=0, background='grey', width=300)
        self.controllerFrame.pack(side='left', fill='y')

        # ------------------------------------------------------------------------

        filebrowser = Frame(self.controllerFrame, borderwidth=3, background='#d9d9d9')
        filebrowser.pack(fill='x', padx=5, pady=5)

        lbl1 = Label(filebrowser, text="1. Press Browse button to select the brainwave file", font=(None, 13), background=LABELBG)
        lbl1.pack(fill='x', pady=5)

        lbl_file = Label(filebrowser, text="Brainwave file: ", font=(None, 13), background='#d9d9d9')
        lbl_file.pack(side='left', pady=5)

        self.pathlabel = Label(filebrowser, background='grey', width=25)
        self.pathlabel.pack(side='left', pady=5)

        button_browse = Button(filebrowser, text="Browse", width=8, command=self.openFile)
        button_browse.pack(side='left', fill='x', padx=10, pady=5)

        # ------------------------------------------------------------------------

        controllPanel = Frame(self.controllerFrame, borderwidth=3, background='#d9d9d9')
        controllPanel.pack(fill='x', padx=5, pady=5)

        # Parameter selection (channal, samplerate)
        #  - Possible values for channal: 1 ~ 38
        #  - Recommended values for samplerate: 200 ~ 600

        lbl2 = Label(controllPanel, text="2. Choose channel and sample rate, then press the Play button", font=(None, 13),
                     background=LABELBG)
        lbl2.pack(fill='x', pady=5)

        channel = Frame(controllPanel)
        channel.pack(fill='x', pady=1)

        lbl_channel = Label(channel, text="Channel", font=(None, 13))
        lbl_channel.pack(side='left')

        self.channal_scaler = Scale(channel, from_=1, to=38, length=300, orient=HORIZONTAL)
        self.channal_scaler.set(1)
        self.channal_scaler.config(borderwidth=2)
        self.channal_scaler.pack(side='right', fill='x', padx=5)

        rate = Frame(controllPanel)
        rate.pack(fill='x', pady=1)

        lbl_sampleRate = Label(rate, text="Sample rate", font=(None, 13))
        lbl_sampleRate.pack(side='left')

        self.sampleRate_scaler = Scale(rate, from_=200, to=600, length=300, orient=HORIZONTAL)
        self.sampleRate_scaler.set(200)
        self.sampleRate_scaler.configure(borderwidth=2)
        self.sampleRate_scaler.pack(side='right', fill='x', padx=5)

        submitButton = Button(controllPanel, text="Play", command=self.melodyFromBrainwave, width=23)
        submitButton.pack(side='left', pady=5)

        stopButton = Button(controllPanel, text="Stop", command=self.stopMusic, width=23)
        stopButton.pack(side='left', fill='x', pady=5)

        # ------------------------------------------------------------------------

        genre_conversion_frame = Frame(self.controllerFrame, background='#d9d9d9')
        genre_conversion_frame.pack(fill='x', padx=5, pady=5)

        lbl3 = Label(genre_conversion_frame, text="3. Select the genre that you want to convert to", font=(None, 13),
                     background=LABELBG)
        lbl3.pack(fill='x', pady=5)

        GENRES = [
            ("Jazz", "jazz"),
            ("Classic", "classic")
        ]

        v = StringVar()
        v.set("jazz") # initalize

        for text, mode in GENRES:
            b = Radiobutton(genre_conversion_frame, text=text, variable=v, value=mode, background='#d9d9d9')
            b.pack(anchor='w', padx=25, pady=3)

        genreButton = Button(genre_conversion_frame, text="Play", width=23, command=self.genreConversion)
        genreButton.pack(side='left', pady=5)

        stopButton = Button(genre_conversion_frame, text="Stop", width=23)
        stopButton.pack(side='left', fill='x', pady=5)

        # Music Frame ----------------------------------------------

        self.musicFrame = Frame(self.parent, relief="solid", borderwidth=1, background="white")
        self.musicFrame.pack(side="right", expand="yes", fill="both")

        lbl4 = Label(self.musicFrame, text="Generated music sheet",
                     font=(None, 13),
                     background=LABELBG)
        lbl4.pack(fill='x', pady=5)

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

    def genreConversion(self):
        pass # will be added later

    def stopMusic(self):
        pass # will be added later

    def melodyFromBrainwave(self):

        # Call brainwave.py module with params to create a melody(in midi format) from brainwave file
        nChannal = self.channal_scaler.get()
        sampleRate = self.sampleRate_scaler.get()

        midiFile = brainwave_to_melody(_filename=self.filePath, _nChannal=nChannal, _sampleRate=sampleRate)

        workingDirPath = os.getcwd()
        midiFilePath = os.path.join(workingDirPath, midiFile)

        # Waiting for the midi file creation
        while True:
            exists = os.path.exists(midiFilePath)
            if exists: break

        # Play the generated midi file
        self.musicPlayer.playMusic(midiFilePath)


if __name__ == "__main__":

    root = Tk()
    root.geometry("1000x400+200+100")

    app = MusicAPP(root)
    root.mainloop()

