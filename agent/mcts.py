import numpy as np
import tensorflow as tf
import time

from graph import *

class Node():
    def __init__(self):
        self.board = np.zeros((1, 2, 15, 15))
        self.player = 0
        self.P = np.zeros(15*15)
        self.N = np.zeros(15*15)
        self.R = np.zeros(15*15)
        self.actions = np.zeros(15*15)
        self.mask = None
        self.children = [None for i in range(15*15)]
        self.win = -1
        self.rollout_res = None
        self.lock = None
        self.prior = None
        
class Net():
    def __init__(self, name):
        graph_dict = AllGraph()
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, name)
        
        self.sess = sess
        self.move_p = graph_dict['policy']['move']
        self.move_r = graph_dict['rollout']['move']
        self.x = graph_dict['policy']['x']
        self.phase = graph_dict['policy']['phase']
        
    def pol_probs(self, board):
        t = time.time()
        ret = self.sess.run(self.move_r, feed_dict={self.x: board, self.phase: False})[0]
        global net_time, net_c
        net_time += time.time()-t
        net_c += 1
        return ret
        
    
    def rol_probs(self, board):
        t = time.time()
        ret = self.sess.run(self.move_r, feed_dict={self.x: board, self.phase: False})[0]
        global net_time, net_c
        net_time += time.time()-t
        net_c += 1
        return ret
    
def check_winner(train_board, player, move):
    t = time.time()
    x = move // 15
    y = move % 15
    board = train_board[0][0] - train_board[0][1]
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for d in directions:
        x_c = x
        y_c = y
        key = 0
        while 0<=x_c-d[0]<15 and 0<=y_c-d[1]<15:
            x_c -= d[0]
            y_c -= d[1]
        while 0<=x_c<15 and 0<=y_c<15:
            if board[x_c][y_c] == 0:
                key == 0
            elif key * board[x_c][y_c] < 0:
                key = board[x_c][y_c]
            else:
                key += board[x_c][y_c]
            if abs(key) == 4:
                if key < 0 and player == 1:
                    return 1
                elif key > 0 and player == 0:
                    return 0
            if abs(key) == 5:
                if key > 0:
                    return 0
                else:
                    return 1
            x_c += d[0]
            y_c += d[1]
    global w_time
    global w_c
    w_time += time.time()-t
    w_c += 1
    return -1
            

def argmax_rand(array):
    return np.random.choice(np.flatnonzero(array == array.max()))

at_time = 0
at_c = 0

w_time = 0
w_c = 0

class GameTree():
    def __init__(self, net):
        self.net = net
        self.root = Node()
        self.root.P = self.net.pol_probs(self.root.board)
        self.root.actions = 10*self.root.P.copy()
        self.check = {'-xxxx':0, 'x-xxx':1, 'xx-xx':2, 'xxx-x':3,
                      'xxxx-':4, '-xxx--':10, '--xxx-':11, '-xx-x-':12,
                      '-x-xx-':13, '-oooo':5, 'o-ooo':6, 'oo-oo':7, 'ooo-o':8,
                      'oooo-':9}
        self.directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        start = [[(i, 0) for i in range(15)], 
                 [(0, i) for i in range(15)]]
        start_diag = [(i, 0) for i in range(11)]
        for i in range(1, 11):
            start_diag.append((0, i))
        start.append(start_diag)
        start_diag = [(14, i) for i in range(11)]
        for i in range(4, 14):
            start_diag.append((i, 0))
        start.append(start_diag)
        self.start = start
        
    def attention(self, state):
        a = time.time()
        me = state.player
        if me == 1:
            other = 0
        else:
            other = 1
        board = state.board[0][me]-state.board[0][other]
        directions = self.directions
        start = self.start
        prior = 18
        prior_dir = None
        x_r = -1
        y_r = -1
        for i in range(4):
            d = directions[i]
            for point in start[i]:
                x = point[0]
                y = point[1]
                ind5 = ''
                ind6 = ''
                while 0<=x<15 and 0<=y<15:
                    if board[x][y] == 0:
                        ind5 += '-'
                        ind6 += '-'
                    elif board[x][y] > 0:
                        ind5 += 'x'
                        ind6 += 'x'
                    else:
                        ind5 += 'o'
                        ind6 += 'o'
                    if len(ind5) == 5:
                        if (ind5 in self.check) and self.check[ind5] < prior:
                            prior = self.check[ind5]
                            prior_dir = d
                            x_r = x
                            y_r = y
                        ind5 = ind5[1:]
                    if len(ind6) == 6:
                        if (ind6 in self.check) and self.check[ind6] < prior:
                            prior = self.check[ind6]
                            prior_dir = d
                            x_r = x
                            y_r = y
                        ind6 = ind6[1:]
                    x += d[0]
                    y += d[1]
        x = x_r
        y = y_r
        d = prior_dir
        if prior == 0:
            x -= d[0]*4
            y -= d[1]*4
        if prior == 1:
            x -= d[0]*3
            y -= d[1]*3
        if prior == 2:
            x -= d[0]*2
            y -= d[1]*2
        if prior == 3:
            x -= d[0]*1
            y -= d[1]*1
        if prior == 10:
            x -= d[0]*1
            y -= d[1]*1
        if prior == 11:
            x -= d[0]*4
            y -= d[1]*4
        if prior == 12:
            x -= d[0]*2
            y -= d[1]*2
        if prior == 13:
            x -= d[0]*3
            y -= d[1]*3
        if prior == 5:
            x -= d[0]*4
            y -= d[1]*4
        if prior == 6:
            x -= d[0]*3
            y -= d[1]*3
        if prior == 7:
            x -= d[0]*2
            y -= d[1]*2
        if prior == 8:
            x -= d[0]*1
            y -= d[1]*1
        global at_time
        global at_c
        at_time += time.time()-a
        at_c += 1
        if prior != 18:
            state.lock = x*15+y
            state.prior = prior
            return True
        else:
            return False
        
    def gen_node(self, parent, move):        
        new_node = Node()
        if parent.player == 0:
            new_node.player = 1
        else:
            new_node.player = 0
        new_node.board = parent.board.copy()
        x = int(move/15)
        y = move%15
        new_node.board[0][parent.player][x][y] = 1
        new_node.P = self.net.pol_probs(new_node.board)
        new_node.win = check_winner(new_node.board, new_node.player, move)
        self.attention(new_node)
        new_node.actions = new_node.P 
        return new_node
    
    def make_rollouts(self, state, depth=12):
        cur_depth = 0
        res = (state.board[0][1]+state.board[0][0]).reshape(15*15)*2
        temp = Node()
        temp.board = state.board.copy()
        temp.player = state.player
        while cur_depth != depth:
            cur_depth += 1
            if self.attention(temp):
                act = temp.lock
            else:
                vov = np.exp(self.net.rol_probs(state.board) - res)
                act = np.random.choice(np.arange(15*15), p=(vov)/vov.sum())
            res[act] -= 2
                
            temp.board[0][temp.player][act//15][act%15] = 1
            if temp.player == 0:
                temp.player = 1
            else:
                temp.player = 0
            win = check_winner(temp.board, temp.player, act)
            if win == 0:
                return 1
            elif win == 1:
                return -1
        return 0
                
    def simulation(self, depth=12, state=None):
        path = []
        if state == None:
            cur = self.root
        else:
            cur = state
        cur_depth = 0
        while cur_depth != depth:
            cur_depth += 1
            if cur.lock != None:
                act = cur.lock
            else:
                res = (cur.board[0][0] + cur.board[0][1]).reshape(15*15)*(1000)
                act = np.argmax(cur.actions-res)
            path.append(act)
            if cur.children[act] == None:
                cur.children[act] = self.gen_node(cur, act)
            cur = cur.children[act]
            if cur.win != -1:
                break
        if cur.win != -1:
            if cur.win == 0:
                reward = 1
            else:
                reward = -1
        else:
            reward = self.make_rollouts(cur, depth)
        if state == None:
            cur = self.root
        else:
            cur = state
        for turn in path:
            cur.N[turn] += 1
            if cur.player == 0:
                cur.R[turn] += reward
            else:
                cur.R[turn] -= reward
            cur.actions[turn] = (cur.R[turn]+10*cur.P[turn])/(cur.N[turn]+1)
            cur = cur.children[turn]
            
        return path, reward

    def move_root(self, turn):
        temp = self.root
        if self.root.children[turn] == None:
            self.root.children[turn] = self.gen_node(self.root, turn)
        self.root = self.root.children[turn]
        del temp