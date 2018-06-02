import numpy as np
import tensorflow as tf

def count_params(graph, print_shapes=False):
    with graph.as_default():
        total_parameters = 0
        params = []
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            if print_shapes:
                print(shape)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            params.append(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)
        print(params)

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