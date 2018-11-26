'''
Author: Zhiyu Yang
Last modified: 11/26/2018
This moudle converts 2D array from network output into midi file
	
	takes a n*m 2D mat as input, where n denotes the number of notes and m denotes range of pitch + duration and channel
	takes the output filename as input 
	it wirtes a midi music to file as well as returns a midi object
	since the input midi for training is actually rescaled when going through midi2mat.py, the input for midi2mat.py and the output for this script are nor exact the same
	for some reason the output of this script is always played by some weird instruments on my pc, it doesn't sound good from my side. 
	But this should depend on the software and setting of your midi player
	
	-new update with piano_roll2midi function (convert piano roll format output to midi file)
	-takes np array (piano roll mat),sample rate(fs) and output filename as input
	-returns midi object and write midi to file
	
'''

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


def piano_roll2midi(mat,outfile,fs):
	import numpy as np
	from midiutil.MidiFile import MIDIFile
	out_midi = MIDIFile(1)
	track = 0
	channel = 0
	time = 0
	tempo = 120
	out_midi.addTrackName(track, time, "output track")
	out_midi.addTempo(track, time, tempo)
	mat = np.hstack((np.zeros([len(mat),1]),mat,np.zeros([len(mat),1])))
	delta_vol = np.diff(mat)
	[a,b] = np.nonzero(delta_vol)
	for i in range(0,len(a)):
		n = a[i]
		m = b[i]
		if delta_vol[n][m] > 0:
			pitch = int(n+1)
			if m > 0 and delta_vol[n][m-1] > 0:
				vol = int(delta_vol[n][m]+delta_vol[n][m-1])
			else:
				vol = int(delta_vol[n][m])
			tStart = m*1/fs
		else:
			if pitch == n+1:
				tEnd = m*1/fs
				duration = tEnd-tStart
			else:
				pitch = int(n+1)
				vol = int(0)
				tStart = m*1/fs
				duration = 0
			out_midi.addNote(track, channel, pitch, tStart, duration, vol)
	filename = "../output/"+outfile+".mid"
	with open(filename, 'wb') as file:
	    out_midi.writeFile(file)
	return(out_midi)




