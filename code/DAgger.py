import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
from random import randint
import numpy as np
import algo_utils
import MaxMI_Algo

n_action = 1        # steer only (float, left and right 1 ~ -1)
steps = 27        # maximum number of questions
n_episode = 5
n_products = number_product()    #TO DO!!!

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "/data/sentences.eval", "Path to the test data. This data should be distinct from the training data.")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#choose the best training checkpoint as a pretrained net
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523379413/checkpoints/", "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS



def get_next_question(state):
    """ Compute the true next question, according to entropy principle, given the history of previous questions and answers.
    Args:
        state: [[question1, answer1],[question2, answer2], ...]
    Returns:
        next_question and boolean variable done
    """
    product_set, traffic_set, purchased_set = get_products(state)                                           #TO DO!!
    question_set_new = set(algo_utils.get_filters_remaining(product_set))
    question_set = question_set_new.difference(state["questions"])
    if len(product_set) < threshold :
        done = True
        next_question = 0
    done = False
    next_question = MaxMI_Algo.opt_step(question_set, product_set, traffic_set, purchased_set)
    #done =True means the remain product_set is smaller than threshold
    return next_question,done

#to add option to choose whether to run MaxMI again or not (if they change catalogue)
if __name__=='__main__':
    state_list = []
    question_list = []
    #reward_list = []

    ###================= Get demonstration data
    """TO USE IN PRE-TRAINING"""
    print("#"*50)
    print('Collecting data from teacher (MaxMI) \n')

    try:
        print('Data found')
        data = tl.files.load_npy_to_any(name='_tmp.npy')
        state_list = data['s']
        question_list = data['q']

    except:
        print("Data not found, asking the teacher to create it \n")
        state_list, question_list = get_data_from_teacher()                       #TO DO!!
        #for all products run MaxMI and get the set of (state, question) it made
        print('Saving data')
        tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': action_list, name = '_tmp.npy')

    ###===================== Pretrain model using data for demonstration
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)


    ###===================== Aggregate and retrain
    output_file = open('results.txt', 'w')
    for episode in range(n_episode):
        q_list = []
        # restart the game for every episode
        print("#" * 50)
        print("# Episode: %d start" % episode)
        y = randint(1, n_products)
        state = []
        while True:
            q = model.predict(state)     #test the model for input state
            a = get_answer(q, y)            #TO DO!

            state.append([q, a])
            q_true, done = get_next_question(state)
            if done is True:
             break
            else:
             q_list.append(q_true)

        print("#" * 50)
        output_file.write('Episode: %02d\t Number or questions: %02d\n' % (episode, len(state)))

        state_list.append(state)
        question_list.append(q_list)

        model.train(state_list, action_list, n_epoch=n_epoch, batch_size=batch_size)  #TO DO !!
        model.save_model()

    ###=================== Play the game with the trained model

"""
###================= Define model
class Agent(object):
    def __init__(self, name='model', sess=None):
        assert sess != None
        self.name = name
        self.sess = sess

        self.x = tf.placeholder(tf.float32, [None], name='States')
        self.y = tf.placeholder(tf.float32, [None], name='Action')

        self._build_net(True, False)
        self._build_net(False, True)
        self._define_train_ops()

        tl.layers.initialize_global_variables(self.sess)

        print()
        self.n_test.print_layers()
        print()
        self.n_test.print_params(False)
        print()
        # exit()

    def _build_net(self, is_train=True, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            n = InputLayer(self.x, name='in')

            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/1')
            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max1')

            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop1')

            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/1')
            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max2')
            # print(n.outputs)
            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop2')

            n = FlattenLayer(n, name='f')
            n = DenseLayer(n, 512, tf.nn.relu, name='dense1')
            n = DropoutLayer(n, 0.5, is_fix=True, is_train=is_train, name='drop3')
            n = DenseLayer(n, n_action, tf.nn.tanh, name='o')

        if is_train:
            self.n_train = n
        else:
            self.n_test = n

    def _define_train_ops(self):
        self.cost = tl.cost.mean_squared_error(self.n_train.outputs, self.y, is_mean=False)
        self.train_params = tl.layers.get_variables_with_name(self.name, train_only=True, printable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost, var_list=self.train_params)

    def train(self, X, y, n_epoch=100, batch_size=10, print_freq=20):
        for epoch in range(n_epoch):
            start_time = time.time()
            total_err, n_iter = 0, 0
            for X_, y_ in tl.iterate.minibatches(X, y, batch_size, shuffle=True):
                _, err = self.sess.run([self.train_op, self.cost], feed_dict={self.x: X_, self.y: y_})
                total_err += err
                n_iter += 1
            if epoch % print_freq == 0:
                print("Epoch [%d/%d] cost:%f took:%fs" % (epoch, n_epoch, total_err/n_iter, time.time()-start_time))

    def predict(self, state):
        a = self.sess.run(self.n_test.outputs, {self.x : state})
        return a

    def save_model(self):
        tl.files.save_npz(self.n_test.all_params, name=self.name+'.npz', sess=self.sess)

    def load_model(self):
        tl.files.load_and_assign_npz(sess=self.sess, name=self.name+'.npz', network=self.n_test)

###===================== Pretrain model using data for demonstration
sess = tf.InteractiveSession()
model = Agent(name='model', sess=sess)
model.train(images_all, actions_all, n_epoch=n_epoch, batch_size=batch_size)
# save model after pretraining
model.save_model()
# model.load_model()
output_file = open('results.txt', 'w')

###===================== Aggregate and retrain
n_episode = 5
for episode in range(n_episode):
    ob_list = []
    # restart the game for every episode
    reward_sum = 0.0
    print("#"*50)
    print("# Episode: %d start" % episode)
    y = randint(1,n_products)
    state = initial_state
    for i in range(steps):
        q = model.predict(state)
        a = get_answer(q, y)
        state.append([q,a])
        ob, done = get_next_question(state)
        if done is True:
            break
        else:
            ob_list.append(ob)

    print("#"*50)
    output_file.write('Number of Steps: %02d\t Number or questions: %02d\n' % (i, len(state)))

    if i == (steps-1):
        break

    state_list.append(state)
    action_list.append(ob_list)

    model.train(state_list, action_list, n_epoch=n_epoch, batch_size=batch_size)
    model.save_model()


"""