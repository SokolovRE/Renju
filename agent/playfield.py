import numpy as np
import os
import sys
import time

im_size = 15

place = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 
         'g':6, 'h':7, 'j':8, 'k':9, 'l':10, 'm':11, 
         'n':12, 'o':13, 'p':14}
letters = ['a', 'b', 'c', 'd', 'e', 'f', 
           'g', 'h', 'j', 'k', 'l', 'm', 
           'n', 'o', 'p']
for i in range(15):
    place[i] = letters[i]

class Playground:
    def __init__(self, black, white):
        self.O = white
        self.X = black
        self.board = np.zeros((15,15))
        self.history = []
        self.history_str = ""
        self.verdict = None
        self.last_move = "\n"
        
    def upd_hist_str(self, player, x, y):
        self.history_str += "  "+player+"-"+place[x]+str(y+1)
        
    def is_over(self):
        board = self.board
        for i in range(11):
            for j in range(15):
                key = board[i][j]
                if key != 0:
                    flag = True
                    for k in range(5):
                        if board[i+k][j] != key:
                            flag = False
                    if flag:
                        return True
        for i in range(15):
            for j in range(11):
                key = board[i][j]
                if key != 0:
                    flag = True
                    for k in range(5):
                        if board[i][j+k] != key:
                            flag = False
                    if flag:
                        return True
        for i in range(11):
            for j in range(11):
                key = board[i][j]
                if key != 0:
                    flag = True
                    for k in range(5):
                        if board[i+k][j+k] != key:
                            flag = False
                    if flag:
                        return True
        for i in range(4, 15):
            for j in range(11):
                key = board[i][j]
                if key != 0:
                    flag = True
                    for k in range(5):
                        if board[i-k][j+k] != key:
                            flag = False
                    if flag:
                        return True
        return False
        
    def paint(self):
        os.system('clear')
        print("    1   2   3   4   5   6   7   8" +
              "   9  10  11  12  13  14  15 ")
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 
                   'g', 'h', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p']
        board = self.board
        for i in range(15):
            print("  "+"-"*61)
            print("%s |" %(letters[i]), end="")
            for j in range(15):
                if (len(self.history) > 0 and
                    i == self.history[-1][0] and
                    j == self.history[-1][1]):
                    if (board[i][j] == -1):
                        print("|O||", end="")
                    elif (board[i][j] == 1):
                        print("|X||", end="")
                    elif (board[i][j] == -2):
                        print("|E||", end="")
                else:
                    if (board[i][j] == 0):
                        print("   |", end="")
                    elif (board[i][j] == -1):
                        print(" O |", end="")
                    else:
                        print(" X |", end="")
                if (j == 14):
                    print("")
        print("  "+"-"*61)
        if (len(self.history) > 0):
            print("\n  History:")
            print(self.history_str)
            print("  "+"-"*61)
        if (self.verdict != None):
            print(self.verdict)
        sys.stdout.flush()
        
    def start_game(self):
        self.paint()
        next_p = 'X'
        while (self.verdict == None):
            if next_p == 'X':
                print('  X turn: ', end='')
                sys.stdout.flush()
                x, y = self.X.move(self)
                self.history.append((x, y))
                if x == -1:
                    self.verdict = '  X gave up. Winner: O'
                    self.paint()
                    break
                self.last_move = place[x]+str(y+1)
                self.upd_hist_str(next_p, x, y)
                if (self.board[x][y] != 0):
                    self.board[x][y] = -2
                    self.verdict = '  X made illegal move. Winner: O'
                else:
                    self.board[x][y] = 1
                    if (self.is_over()):
                        self.verdict = '  Winner: X';
                next_p = 'O'
            else:
                print('  O turn: ', end='')
                sys.stdout.flush()
                x, y = self.O.move(self)
                self.history.append((x, y))
                if x == -1:
                    self.verdict = '  O gave up. Winner: X'
                    self.paint()
                    break
                self.last_move = place[x]+str(y+1)
                self.upd_hist_str(next_p, x, y)
                if (self.board[x][y] != 0):
                    self.board[x][y] = -2
                    self.verdict = '  O made illegal move. Winner: X'
                else:
                    self.board[x][y] = -1
                    if (self.is_over()):
                        self.verdict = '  Winner: O'
                next_p = 'X'
            if self.verdict == None and len(self.history) == 15*15:
                self.verdict = 'Draw'
            self.paint()
        
class Human():
    def move(self, playground):
        turn = sys.stdin.readline()
        try:
            x = place[turn[0]]
            y = int(turn[1:])-1
        except:
            return -1, -1
        if 0<=x<15 and 0<=y<15:
            return x, y
        else:
            return -1, -1
        
class Agent():
    def __init__(self, game_tree, sec=15):
        self.tree = game_tree
        self.sec = sec
    
    def move(self, playground):
        print('Please wait.', end='')
        sys.stdout.flush()
        turn = playground.last_move
        if turn != "\n":
            turn = 15*place[turn[0]]+int(turn[1:])-1
            self.tree.move_root(turn)
            
        time_begin = time.time()
        while time.time()-time_begin < self.sec and self.sec >= 10:
            self.tree.simulation()
        if self.sec < 10:
            turn = np.argmax(self.tree.root.P)
        else:
            turn = np.argmax(self.tree.root.N)
        if self.tree.root.lock != None:
            turn = self.tree.root.lock
        self.tree.move_root(turn)
        x = turn//15
        y = turn%15
        return x, y