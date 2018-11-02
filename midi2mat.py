'''
Author: Zhiyu Yang
Last modified: 11/02/2018

This moudle is for converting midi file to mat

	- It takes parameters:

	1. mode(1 for brainwave midi transformation,2 for regular midi file transformation)
	2.for mode 1,specify channel,sample rate (to construct input filename),seperate by comma; for mode 2, specify input filename
	3&4.lower/upper bound (int) for pitch coding (default 1-88 if not provided)

	- tested and works for brainwave-midi, need verification for regular midi
	- tested for regular midi, works but a little weird
	- for some reason some of the note durations are 0 in the midi file
'''

import sys
import numpy as np
#from music21 import midi

############fancy stuff##################

if sys.argv[1] == "1":
	a,b = sys.argv[2].split(",")
	filename = "Brainwave_channal" + a + "_SR" + b + ".mid"
elif sys.argv[1] == "2":
	filename = sys.argv[2]
else:
	print("specify transforation mode")

if sys.argv[3] != None:
	lower = int(sys.argv[3])
else:
	lower = 1

if sys.argv[4] != None:
	upper = int(sys.argv[4])
else:
	upper = 88

##############useful stuff#################
def midi2mat(filename,upper=88,lower=1):
	import numpy as np
	from music21 import midi  #just in case
	file = midi.MidiFile()
	file.open(filename,attrib='rb')
	file.read()
	file.close()   #read midi file and decode
	MDtrack = []   #translate midifile into midi track
	for i in range(0,len(file.tracks)):
		if file.tracks[i].hasNotes():
			MDtrack.append(file.tracks[i])  
	INFO = []   #translate midi track into midi events and write into arrays
	for i in range(0,len(MDtrack)):
		MDevent = MDtrack[i].events
		info = {"pitch":[],"duration":[],"velocity":[],"channel":[]}   #dictionary for one track of information, incase there are multiple
		for j in range(0,len(MDevent)):
			if MDevent[j].type == "NOTE_ON":
				info["pitch"].append(MDevent[j].pitch)
				info["velocity"].append(MDevent[j].velocity)  #velocity.. I guess indicating volume..?
				info["channel"].append(MDevent[j].channel)
				if MDevent[j+1].type == "DeltaTime":  #seem like the regular midi file doesn't have event NoteOff
					info["duration"].append(MDevent[j+1].time)
				else:
					print("WARNING: track "+str(i)+", NoteOn event "+str(j))
		x = np.subtract(info["pitch"],min(info["pitch"]))/(max(info["pitch"])-min(info["pitch"]))
		x = list(map(round,np.multiply(x,upper-lower)+1))
		info["pitch"] = x  #rescale the data and linearly map onto the pitch coding range. need to think about this
		INFO.append(info)
	OUTPUT = []
	for i in range(0,len(INFO)):
		layer = np.zeros((len(INFO[i]["pitch"]),upper+2))
		for j in range(0,len(INFO[i]["pitch"])):
			layer[j][int(INFO[i]["pitch"][j])-1] = INFO[i]["velocity"][j]
			layer[j][upper] = INFO[i]["duration"][j]
			layer[j][upper+1] = INFO[i]["channel"][j]
		OUTPUT.append(layer)
	return(OUTPUT)


#########if run the whole script, output a txt file for EACH TRACK#########
#highly not recommended, works but so ugly
output = midi2mat(filename,upper,lower)
for i in range(0,len(output)):
	output_name = filename + "track_" + str(i+1)
	np.savetxt(output_name,output[i])







