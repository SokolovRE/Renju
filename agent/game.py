#! /usr/bin/python3.4
import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from graph import *
from mcts import *
from playfield import *

if __name__ == "__main__":
    print('Enter path to weights (Default is \"./contest/allgraph\"): ', end='')
    sys.stdout.flush()
    weights_path = sys.stdin.readline()
    if weights_path == "\n":
        weights_path = "./contest/allgraph"
    else:
        weights_path = weights_path[:-1]
    try:
        net = Net(weights_path)
    except:
        print('Cannot load weights.')
        exit()
    game_tree = GameTree(net)
        
    human_p = ""
    while human_p != "O\n" and human_p != "X\n" and human_p != "\n":
        print('Choose player ([X]/O): ', end='')
        sys.stdout.flush()
        human_p = sys.stdin.readline()
        if human_p == "O\n":
            print('Enter time for one move in sec (Default 15, if less then 10 agent will use only network to play): ', end='')
            sys.stdout.flush()
            sec = sys.stdin.readline()
            if sec == "\n":
                sec = 15
            else:
                sec = float(sec)
            human = Human()
            agent = Agent(game_tree, sec)
            game = Playground(agent, human)
        elif human_p == "\n" or human_p == "X\n":
            print('Enter time for one move in sec (Default 15, if less then 10 agent will use only network to play): ', end='')
            sys.stdout.flush()
            sec = sys.stdin.readline()
            if sec == "\n":
                sec = 15
            else:
                sec = float(sec)
            human = Human()
            agent = Agent(game_tree, sec)
            game = Playground(human, agent)
    game.start_game()