'''
    Author: Hyeonae Jang
    Last modified: 11/01/2018
    This moudle is to generate pasrsed form of notes and chord of seed(brainwave) music
    '''

import pickle
from music21 import converter, instrument, note, chord

def get_input_notes():
    notes=[]

    file="Brainwave_channal1_SR500.mid"
    midi = converter.parse(file)
    # excerpt from a midi file that has been read using Music21
    midi.show('text')

    print("Parsing %s" % file)

    #brainwave music file has notes in a flat structure
    notes_to_parse = midi.flat.notes
    
    for element in flat_notes:
        #parse notes
        if isinstance(element, notes_to_parse):
            notes.append(str(element.pitch))
        #parse chord
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    filepath=open('data/notes', 'wb')
    pickle.dump(notes, filepath)

    #a list of parsed notes and chord
    return notes

#get_input_notes()
