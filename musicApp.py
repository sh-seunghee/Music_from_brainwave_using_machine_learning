'''
Author: Seunghee Lee
Last modified: 11/29/2018

We use python library TKinter to provide a graphical user interface.

The module contains following components:

    1. Inputs a recorded brainwave file
    2. Gets an input of a channel and a sample rate value from the user
    3. Plays a reconstructed melody from the brainwave file
    4. User selects music genre that they want to convert into
    5. Plays the genre-converted melody
    6. Shows the music score

'''
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from subprocess import Popen
from PIL import Image, ImageTk
from musicPlayer import *
from brainwave2midi import brainwave_to_melody
from music_transfer import to_transfer
from brainMIDI_modifier import modify_music

class MusicAPP():

    def __init__(self, parent):

        super().__init__()

        self.parent = parent
        self.LABELBG = '#b5b5b5'                    # The color commonly used for labels in this GUI

        self.bw_filePath = None                     # brainwave file path
        self.midiFilePath = None                    # midi file path
        self.user_selected_genre = StringVar()      # User-selected genre

        self.musicPlayer = MusicPlayer()            # Create an initialize an instance of the MusicPlayer class

        self.initUI()                               # initialize the UI

    def initUI(self):
        # Set window title
        self.parent.title("Brainwave to Music Conversion Software")

        # Controller Frame -------------------------------------------------------
        self.controllerFrame = Frame(self.parent, borderwidth=0, background='grey', width=300)
        self.controllerFrame.pack(side='left', fill='y')

        # Sub-frame: File broswer ------------------------------------------------
        filebrowser = Frame(self.controllerFrame, borderwidth=3, background='#d9d9d9')
        filebrowser.pack(fill='x', padx=5, pady=5)

        lbl1 = Label(filebrowser, text="1. Press Browse button to select the brainwave file", font=(None, 13), background=self.LABELBG)
        lbl1.pack(fill='x', pady=5)

        lbl_file = Label(filebrowser, text="Brainwave file: ", font=(None, 13), background='#d9d9d9')
        lbl_file.pack(side='left', pady=5)

        self.pathlabel = Label(filebrowser, background='grey', width=25)
        self.pathlabel.pack(side='left', pady=5)

        button_browse = Button(filebrowser, text="Browse", width=8, command=self.openbwFile)
        button_browse.pack(side='left', fill='x', padx=10, pady=5)

        # Sub-frame: Control panel ------------------------------------------------
        controlPanel = Frame(self.controllerFrame, borderwidth=3, background='#d9d9d9')
        controlPanel.pack(fill='x', padx=5, pady=5)

        # Parameter selection (channal, samplerate)
        #  - Possible values for channal: 1 ~ 38
        #  - Recommended values for sample rate: 200 ~ 600

        lbl2 = Label(controlPanel, text="2. Choose channel and sample rate, then press the Play button", font=(None, 13), background=self.LABELBG)
        lbl2.pack(fill='x', pady=5)

        channel = Frame(controlPanel)
        channel.pack(fill='x', pady=1)

        lbl_channel = Label(channel, text="Channel", font=(None, 13))
        lbl_channel.pack(side='left')

        self.channal_scaler = Scale(channel, from_=1, to=38, length=300, orient=HORIZONTAL)
        self.channal_scaler.set(1)
        self.channal_scaler.config(borderwidth=2)
        self.channal_scaler.pack(side='right', fill='x', padx=5)

        rate = Frame(controlPanel)
        rate.pack(fill='x', pady=1)

        lbl_sampleRate = Label(rate, text="Sample rate", font=(None, 13))
        lbl_sampleRate.pack(side='left')

        self.sampleRate_scaler = Scale(rate, from_=200, to=600, length=300, orient=HORIZONTAL)
        self.sampleRate_scaler.set(200)
        self.sampleRate_scaler.configure(borderwidth=2)
        self.sampleRate_scaler.pack(side='right', fill='x', padx=5)

        submitButton = Button(controlPanel, text="Play", command=self.melodyFromBrainwave, width=23)
        submitButton.pack(side='left', pady=5)

        stopButton = Button(controlPanel, text="Stop", command=self.stopMusic, width=23)
        stopButton.pack(side='left', fill='x', pady=5)

        # Sub-frame: Genre conversion frame ---------------------------------------
        genre_conversion_frame = Frame(self.controllerFrame, background='#d9d9d9')
        genre_conversion_frame.pack(fill='x', padx=5, pady=5)

        lbl3 = Label(genre_conversion_frame, text="3. Select the genre that you want to convert to", font=(None, 13), background=self.LABELBG)
        lbl3.pack(fill='x', pady=5)

        GENRES = [
            ("Jazz", "jazz"),
            ("Classical", "classical")
        ]

        for text, mode in GENRES:
            b = Radiobutton(genre_conversion_frame, text=text, variable=self.user_selected_genre, value=mode, background='#d9d9d9')
            b.pack(anchor='w', padx=25, pady=3)

        self.user_selected_genre.set("jazz")

        genreButton = Button(genre_conversion_frame, text="Play", width=23, command=self.genreConversion)
        genreButton.pack(side='left', pady=5)

        stopButton = Button(genre_conversion_frame, text="Stop", command=self.stopMusic, width=23)
        stopButton.pack(side='left', fill='x', pady=5)

        # Music score frame -----------------------------------------------------
        self.musicScoreFrame = Frame(self.parent, relief="solid", borderwidth=1, background="white")
        self.musicScoreFrame.pack(side="right", expand="yes", fill="both")

        lbl4 = Label(self.musicScoreFrame, text="Generated music sheet", font=(None, 14), background=self.LABELBG)
        lbl4.pack(fill='x', pady=10)

        self.musicScore_canvas = Canvas(self.musicScoreFrame)
        self.musicScore_canvas.pack(side=LEFT, expand=YES, fill=BOTH)

    def openbwFile(self):
        # Open the brainwave file
        self.bw_filePath = askopenfilename(title="Choose a brainwave file")

        print("Open brainwave file: " + str(self.bw_filePath))
        self.pathlabel.config(text=self.bw_filePath)

    def genreConversion(self):
        if self.midiFilePath is None:
            messagebox.showinfo("ALERT", "Please play the brainwave file first")

        else:
            # Clear the canvas
            self.musicScore_canvas.delete("all")

            # Call to_transfer in music_transfer.py for genre conversion
            classic_fname, jazz_fname = to_transfer(self.midiFilePath, G_AB_classical_1="data/G_AB_classical.pth", G_AB_jazz_1="data/G_AB_jazz.pth")

            # Play the generated midi file
            if self.user_selected_genre.get() == "jazz":
                print ("Genre converted to Jazz...")
                self.genre_converted_music_filePath = "output/"+jazz_fname+".mid"

            elif self.user_selected_genre.get() == "classical":
                print ("Genre converted to Classical..")
                self.genre_converted_music_filePath = "output/" + classic_fname + ".mid"

            # generate the corresponding music score on the right frame
            self.generateMusicScore(inputMidi=self.genre_converted_music_filePath)

            # show the music score on the music score frame
            score_img = Image.open("output/score-1.png")
            photoImg = ImageTk.PhotoImage(score_img)
            self.musicScore_canvas.create_image(0, 0, anchor="nw", image=photoImg)

            # Play the genre converted music
            self.musicPlayer.playMusic(self.genre_converted_music_filePath)

            self.musicScore_canvas.mainloop()

    def stopMusic(self):
        self.musicPlayer.stopMusic()

    def generateMusicScore(self, inputMidi):
        # Spawn a new process of generating the music score from the .mid input
        generate_score_process = Popen(
            ['/Applications/MuseScore 2.app/Contents/MacOS/mscore', '-I', inputMidi, '-o',
             'output/score.png', '-r', '100'])

        # Interact with the process
        stdout, stderr = generate_score_process.communicate()

        # Check err if any
        if stderr is not None:
            print(stderr)

    def melodyFromBrainwave(self):
        # Get input values from the user
        nChannal = self.channal_scaler.get()
        sampleRate = self.sampleRate_scaler.get()

        try:
            # Clear the canvas
            self.musicScore_canvas.delete("all")

            # Call brainwave2midi.py module with the params to create a melody(in midi format) from the brainwave file
            self.midiFilePath = brainwave_to_melody(_filename=self.bw_filePath, _nChannal=nChannal, _sampleRate=sampleRate)
            
            # Call brainMIDI_modifier.py module to modify and generalize brainwave music 
            #modify_music(_filepath=self.midiFilePath)
            

            # generate the corresponding music score on the right frame
            self.generateMusicScore(inputMidi=self.midiFilePath)

            # show the music score on the music score frame
            score_img = Image.open("output/score-1.png")
            photoImg = ImageTk.PhotoImage(score_img)
            self.musicScore_canvas.create_image(0, 0, anchor="nw", image=photoImg)

            # Play the generated midi file
            self.musicPlayer.playMusic(self.midiFilePath)

            self.musicScore_canvas.mainloop()



        except:
            if self.bw_filePath is None:
                messagebox.showinfo("ALERT", "No brainwave file is inserted!")
            else:
                messagebox.showinfo("ALERT","Wrong file selected:\n" + str(self.bw_filePath) + "\nPlease select the midi formatted file!")

if __name__ == "__main__":

    root = Tk()
    root.geometry("1300x2000+100+100")

    app = MusicAPP(root)
    root.mainloop()
