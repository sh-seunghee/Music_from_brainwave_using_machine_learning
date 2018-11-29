    '''
    Author: Hyeonae Jang
    Last modified: 11/29/2018
    This moudle is to generate parsed notes and chord of music(midi file),
    normalize the pitch(octave), convert chord with major chords, and adjust offset(distance between notes)
    as a way of generalizing and stylizing raw music
    '''

import glob
import pickle
import random
from music21 import converter, instrument, note, chord, stream

def modify_music():
    initialNotes = get_notes()
    normalizedNotes = normalize_pitch(initialNotes)
    majorNotes = major_chord(normalizedNotes)
    print(majorNotes)
    # create_cycle(majorNotes)
    create_midi(majorNotes)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi directory """
    notes = []

    for file in glob.glob("midi/*.mid"):
        midi = converter.parse(file)

        # excerpt from a midi file that has been read using Music21
        midi.show('text')

        # music file has notes in a single instrument
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    print("notes",len(notes),notes)

    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    # notes consist of note and chord
    return notes

def normalize_pitch(notes):

    pitch=[]
    normalizedNotes=[]
    # all the octav from note
    pitch = [note[-1] for note in notes if not ('.' in note) or note.isdigit()]
    # most frequent octav
    #octavCnt=Counter(octav)
    pitchCounts = {p:pitch.count(p) for p in pitch}
    pitchCounts= sorted(pitchCounts.items(), key=lambda x: x[1],reverse=True)

    freq_pitch=[]
    #find 2 most frequent pitch
    for p in range(0,2):
        freq_pitch.append(pitchCounts[p][0])

    #sorted freq_pitch by converting to int
    int_freq_pitch=sorted(list(map(int, freq_pitch)))

    for i in range(len(notes)):# for note in notes: #--> immutable
        # in case of a note
        if any(j.isalpha() for j in notes[i]):
            # if the octav is too high or too low
            if not notes[i][-1] in freq_pitch:
                # increase the octav
                if int(notes[i][-1]) < int_freq_pitch[0]:
                    notes[i] = notes[i][:-1] + str(int_freq_pitch[0])
                # decrease the octav
                else:
                    notes[i] = notes[i][:-1]+ str(int_freq_pitch[1])
    normalizedNotes=notes
    return normalizedNotes

def major_chord(normalizedNotes):
    majorNotes=[]

    #Major chords in all 12 keys
    majorChord={'0':'0.4.7','1':'1.5.8','2':'2.6.9','3':'3.7.10','4':'4.8.11','5':'5.9.0',
                '6':'6.10.1','7':'7.11.2','8':'8.0.3','9':'9.1.4','10':'10.2.5','11':'11.3.6'}
    i=0
    while(i<len(normalizedNotes)):
        # notes is a chord
        if ('.' in normalizedNotes[i]):
            notes_in_chord=normalizedNotes[i].split('.')
            del normalizedNotes[i]
            for j in range(len(notes_in_chord)):
                for key,value in majorChord.items():
                    if notes_in_chord[j] == key:
                        #replace each note in chord with major chords
                        notes_in_chord[j]= value

            #inserts major chords inbetween of notes
            normalizedNotes[i:i]=notes_in_chord

            #revise iteration flag due to the added major chords
            i+=len(notes_in_chord)
            #print(notes_in_chord)
        else:
            i+=1

    majorNotes=normalizedNotes
    return majorNotes

def create_cycle(notes):
    key = []

    keyDict={'C':0,'C#':1,'D':2,'E-':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'B-':10,'B':11}
    # all the keys from note
    key = [note[:-1] for note in notes if not ('.' in note) or note.isdigit()]
    #print(key)

    for i in range(len(key)):
        for k, v in keyDict.items():
            if key[i]==k:
                key[i]=v
    #key.remove('')
    #print(key)

def create_midi(outputs):
    """ convert the output to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects
    for output in outputs:
        # output is a chord
        if ('.' in output) or output.isdigit():
            notes_in_chord = output.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # output is a note
        else:
            new_note = note.Note(output)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)


        # increase offset each iteration so that notes do not stack
        offset += 0.5 #random.uniform(0.1,1.0)

    print(output_notes)
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='output.mid')

if __name__ == '__main__':
    modify_music()

