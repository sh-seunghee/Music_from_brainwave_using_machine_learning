'''
    Author: Hyeonae Jang
    Last modified: 11/01/2018
    This moudle is to generate parsed form of notes and chord of music(midi file)
    '''

import pickle
from music21 import converter, instrument, note, chord

def midi_to_notes():
    notes=[]
    
    file="Brainwave_channal1_SR500.mid" #"Bymyself.mid"
    midi = converter.parse(file)
    # excerpt from a midi file that has been read using Music21
    midi.show('text')
    
    print("Parsing %s" % file)
    
    #music file has notes in a single instrument
    notes_to_parse = midi.flat.notes
    
    for element in notes_to_parse:
        #parse notes
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        #parse chord
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    print(notes)
    filepath=open('data/notes', 'wb')
    pickle.dump(notes, filepath)

    #a list of parsed notes and chord
    return notes
