#! /usr/bin/python3.4
import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf

games_num = 1984694

move_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 
             'g':6, 'h':7, 'j':8, 'k':9, 'l':10, 'm':11, 
             'n':12, 'o':13, 'p':14}
letters = ['a', 'b', 'c', 'd', 'e', 'f', 
           'g', 'h', 'j', 'k', 'l', 'm', 
           'n', 'o', 'p']
for i in range(15):
    move_dict[i] = letters[i]
    
net_time = 0
net_c = 0
    
def AllGraph():
    def Conv2D(filters, kernel=5, name=None):
        return tf.layers.Conv2D(filters, kernel, padding='same', name=name)

    def BatchNorm(name=None):
        return tf.keras.layers.BatchNormalization()
    
    p_dict = {}
    r_dict = {}
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape=[None, 2, 15, 15])
    y_ = tf.placeholder(tf.int64)
    phase = tf.placeholder(tf.bool)
    alpha = tf.placeholder(tf.float32)
    p_dict['x'] = x
    p_dict['y_'] = y_
    p_dict['phase'] = phase
    p_dict['alpha'] = alpha
    r_dict['x'] = x
    r_dict['y_'] = y_
    r_dict['phase'] = phase
    r_dict['alpha'] = alpha
        
    tran = tf.transpose(x, [0, 2, 3, 1])
        
    model = Conv2D(32)(tran)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(64)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(64)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    out = tf.layers.Flatten()(Conv2D(1, 3)(model))
        
    move = tf.nn.softmax(out)
    p_dict['move'] = move

    check_prediction = tf.equal(tf.argmax(out,1), y_)
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    p_dict['accuracy'] = accuracy

    loss_function = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=out))
    p_dict['loss'] = loss_function

    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_ops):
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss_function)
    p_dict['train_step'] = train_step
            
            
    model = Conv2D(32)(tran)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(32)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    model = Conv2D(64)(model)
    model = BatchNorm()(model, training=phase)
    model = tf.nn.relu(model)
        
    out = tf.layers.Flatten()(Conv2D(1, 3)(model))
        
    move = tf.nn.softmax(out)
    r_dict['move'] = move

    check_prediction = tf.equal(tf.argmax(out,1), y_)
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    r_dict['accuracy'] = accuracy

    loss_function = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=out))
    r_dict['loss'] = loss_function

    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_ops):
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss_function)
    r_dict['train_step'] = train_step
        
    graph = tf.get_default_graph()
    p_dict['graph'] = graph
    r_dict['graph'] = graph
            
    return {'policy': p_dict, 'rollout': r_dict, 'graph': graph}

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
        
LOG_FORMAT = '%(levelname)s:%(asctime)s: retard-{0}: %(message)s'.format(os.getpid())
        
def main():
    pid = os.getpid()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start retarded backend...")
    
    time_begin = time.time()
    
    logging.debug("Loading networks, initializing GameTree.")
    net = Net('./allgraph')
    tree = GameTree(net)
    
    try:
        while True:
            global net_time, net_c, at_time, at_c, w_time, w_c
            game = sys.stdin.readline()
            logging.debug("-------------------------------")
            turn_timer = time.time()
            if not game:
                logging.debug("Game is over!")
                return
            if game != '\n':
                turn = game.split(' ')[-1]
                turn = 15*move_dict[turn[0]]+int(turn[1:])
                tree.move_root(turn)
            counter = 0
            time_begin = time.time()
            count_time = time.time()
            while time.time()-time_begin < 14:
                tree.simulation()
                #logging.debug("%f sec per simulation." %(time.time()-count_time))
                count_time = time.time()
                counter += 1
            logging.debug("%d simulations." %(counter))
            if w_c*at_c*net_c != 0:
                logging.debug("%f check winner %d times %f sum." %(w_time/w_c, w_c, w_time))
                logging.debug("%f attention %d times %f sum." %(at_time/at_c, at_c, at_time))
                logging.debug("%f net %d times %f sum." %(net_time/net_c, net_c, net_time))
            net_time = 0
            net_c = 0
            at_time = 0
            at_c = 0
            w_time = 0
            w_c = 0
            turn = np.argmax(tree.root.N)
            if tree.root.lock != None:
                logging.debug("LOCK WORKED %d" %(tree.root.prior))
                turn = tree.root.lock
            tree.move_root(turn)
            turn = move_dict[turn//15] + str(turn%15)
            logging.debug("My turn: %s" %(turn))
            
            if sys.stdout.closed:
                logging.debug("Stdout is closed!")
                return
            logging.debug("%f - turn time." %(time.time()-turn_timer))
            logging.debug("-------------------------------")
            sys.stdout.write(turn + '\n')
            sys.stdout.flush()
            
    except:
        logging.error('Error!', exc_info=True, stack_info=True)
        
if __name__ == "__main__":
    main()
    
