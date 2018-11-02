#Zhiyu Yang
#11/02/2018

#This method transfers 2D arrays output from the network into midi file

#takes a n*m 2D mat as input, where n denotes the number of notes and m denotes range of pitch + duration and channel
#takes the output filename as input 
#it wirtes a midi music to file as well as returns a midi object
#since the input midi for training is actually rescaled when going through midi2mat.py, the input for midi2mat.py and the output for this script are nor exact the same
#for some reason the output of this script is always played by some weird instruments on my pc, it doesn't sound good from my side. 
#But this should depend on the software and setting of your midi player

def mat2midi(mat,name,SampleRate):
	from midiutil.MidiFile import MIDIFile  #incase
	import numpy as np
	[a,b] = mat.shape
	tStart = [0]
	out_midi = MIDIFile(1)
	track = 0
	time = 0
	tempo = 120
	out_midi.addTrackName(track, time, "output track")
	out_midi.addTempo(track, time, tempo)
	nTrack = 1
	for i in range(0,a):
		ind = [x for x in range(0,b-2) if mat[i][x] != 0]
		while len(ind) > nTrack:
			out_midi.addTrackName(nTtrack, time, "output track")
			out_midi.addTempo(nTrack, time, tempo)
			nTrack = nTrack+1
		dur = (mat[i][-2])/SampleRate
		for k in range(0,len(ind)):
			pit = ind[k]
			channel = int(mat[i][-1])
			vol = int(mat[i][ind[k]])
			out_midi.addNote(k, channel, pit, tStart[-1], dur, vol)
		tStart.append(tStart[-1]+dur)
	filename = "../output/"+name+".mid"
	with open(filename, 'wb') as file:
		out_midi.writeFile(file)
	return(out_midi)





