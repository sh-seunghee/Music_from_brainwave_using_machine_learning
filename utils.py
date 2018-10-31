'''

Author: Seunghee Lee
Last modified: 10/31/2018

This module contains functions that are used redundantly for this program.

    - playMusic: This function plays a given music file by using the library 'pygame'

'''

import pygame

def playMusic(musicFilePath):

    '''

    - code references: https://gist.github.com/guitarmanvt/3b6a91cefb2f5c3098ed

    :param musicFilePath: path for midifile
    :return:

    '''

    pygame.init()
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(musicFilePath)
        print ("Music file %s loaded!" % musicFilePath)

    except pygame.error as e:
        print (e)
        print ("File %s not found! (%s)" % (musicFilePath, pygame.get_error()))
        return

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():

        # check if playback has finished
        clock.tick(30)

