"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my page: https://github.com/hnVfly/DRL_mxnet

Using:
python 3.6.1
mxnet 0.10.1
gym 0.9.2
"""
import mxnet as mx
import numpy as np
import gym

np.random.seed(1)
mx.random.seed(1)

MAX_EPISODES = 70
MAX_EP_STEPS = 400
LR_A = 0.01
LR_C = 0.01
GAMMA = 0.9
TAU = 0.01
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32
CTX = mx.cpu()

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

class Actor(object):
    def __init__(self, n_action, n_features, action_bound, learning_rate, t_replace_iter, batchsize=1, ctx=mx.cpu()):
        self.a_dim = n_action
        self.n_features = n_features
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.batchsize = batchsize
        self.ctx = ctx
        self.modA_target = self.createActornetwork()
        self.modA_eval = self.createActornetwork(isTrain=True)
        self.copyTargetANetwork()

    def Actor_sym(self):
        s = mx.sym.Variable('state')
        l1 = mx.sym.FullyConnected(data=s, num_hidden=30, name='l1')
        relu1 = mx.sym.Activation(data=l1, act_type='relu', name='relu1')
        acts_prob = mx.sym.FullyConnected(data=relu1, num_hidden=self.a_dim, name='acts_prob')
        acts_prob_sm = mx.sym.Activation(data=acts_prob, act_type='tanh', name='acts_prob_sm')
        acts_prob_sm = acts_prob_sm * self.action_bound[0]
        return acts_prob_sm

    def createActornetwork(self, isTrain=False):
        if isTrain:
            modA = mx.mod.Module(symbol=self.Actor_sym(), data_names=['state', ], label_names=None, context=self.ctx)
            modA.bind(data_shapes=[('state', (self.batchsize, self.n_features))],
                      label_shapes=None, for_training=isTrain)
            modA.init_params(initializer=mx.init.Normal(.1))
            modA.init_optimizer(optimizer='Adam',
                                optimizer_params={'learning_rate':self.lr})
        else:
            modA = mx.mod.Module(symbol=self.Actor_sym(), data_names=['state', ], label_names=None, context=self.ctx)
            modA.bind(data_shapes=[('state', (self.batchsize, self.n_features))], label_shapes=None,
                      for_training=isTrain)
            modA.init_params(initializer=mx.init.Normal(.1))
        return modA

    def choose_action_(self, s_):
        feed_dict = mx.io.DataBatch([mx.nd.array(s_, ctx=self.ctx)])
        self.modA_target.forward(feed_dict)
        actions_ = self.modA_target.get_outputs()[0].asnumpy()
        return actions_

    def learn(self, s, a_grad):
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.copyTargetANetwork()
        train_feed_dict = mx.io.DataBatch([mx.nd.array(s, ctx=self.ctx)])
        self.modA_eval.forward(train_feed_dict)
        self.modA_eval.backward([-a_grad])
        self.modA_eval.update()
        self.t_replace_counter += 1

    def copyTargetANetwork(self):
        arg_params,aux_params=self.modA_eval.get_params()
        self.modA_target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)
        #print('time to copy modA')

    def choose_action(self, s):
        s = s[np.newaxis,:]
        self.modA_eval.forward(mx.io.DataBatch([mx.nd.array(s,self.ctx)]), is_train=False)
        actions = self.modA_eval.get_outputs()[0].asnumpy()[0]
        return actions

class Critic(object):
    def __init__(self, n_action, n_features, learning_rate, t_replace_iter, batchsize=1, ctx=mx.cpu()):
        self.n_action = n_action
        self.n_features = n_features
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.batchsize = batchsize
        self.ctx = ctx
        self.modC_target = self.createCriticnetwork()
        self.modC_eval = self.createCriticnetwork(isTrain=True)

    def Critic_sym(self, predict=False):
        s = mx.sym.Variable('state')
        a = mx.sym.Variable('action')
        data = mx.sym.concat(s, a, dim=1)
        l1 = mx.sym.FullyConnected(data=data, num_hidden=30, name='l1')
        relu1 = mx.sym.Activation(data=l1, act_type='relu', name='relu1')
        q = mx.sym.FullyConnected(data=relu1, num_hidden=1, name='q')
        if predict:
            return q
        else:
            r = mx.sym.Variable('reward')
            q_ = mx.sym.Variable('q_')
            q_target = r + GAMMA * q_
            error = q - q_target
            td_error = mx.sym.square(error)
            loss = mx.sym.MakeLoss(td_error)
            output = mx.sym.Group([loss, mx.sym.BlockGrad(error), mx.sym.BlockGrad(q)])
            return output

    def createCriticnetwork(self, isTrain=False):
        if isTrain:
            modC = mx.mod.Module(symbol=self.Critic_sym(), data_names=('state', 'action'), label_names=('reward','q_'),context=self.ctx)
            modC.bind(data_shapes=[('state',(self.batchsize, self.n_features)),('action',(self.batchsize,self.n_action))],
                      label_shapes=[('reward',(self.batchsize, 1)), ('q_', (self.batchsize, 1))],
                      for_training=True, inputs_need_grad=True)
            modC.init_params(initializer=mx.init.Normal(0.1))
            modC.init_optimizer(optimizer='Adam',
                                optimizer_params={'learning_rate': self.lr})
        else:
            modC = mx.mod.Module(symbol=self.Critic_sym(predict=True), data_names=('state', 'action'), label_names=None,
                                 context=self.ctx)
            modC.bind(data_shapes=[('state', (self.batchsize, self.n_features)), ('action', (self.batchsize, self.n_action))],
                label_shapes=None, for_training=False)
            modC.init_params(initializer=mx.init.Normal(0.1))
        return modC

    def learn(self, s, a, r, s_, a_):
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.copyTargetCNetwork()
        feed_dict = mx.io.DataBatch([mx.nd.array(s_, self.ctx), mx.nd.array(a_, self.ctx)])
        self.modC_target.forward(feed_dict)
        q_ = self.modC_target.get_outputs()[0].asnumpy()
        train_feed_dict = mx.io.DataBatch([mx.nd.array(s, self.ctx), mx.nd.array(a, self.ctx)],
                                          [mx.nd.array(r, self.ctx), mx.nd.array(q_, self.ctx)])
        self.modC_eval.forward(train_feed_dict)
        error = self.modC_eval.get_outputs()[1].asnumpy()
        self.modC_eval.backward()
        self.modC_eval.update()
        a_grad = self.modC_eval.get_input_grads()[1]
        self.t_replace_counter += 1
        return a_grad/(2 * mx.nd.array(error, self.ctx))

    def copyTargetCNetwork(self):
        arg_params,aux_params=self.modC_eval.get_params()
        self.modC_target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)
        #print('time to copy modC')

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1
    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

actor = Actor(n_action=action_dim, n_features=state_dim, action_bound=action_bound, learning_rate=LR_A, t_replace_iter=REPLACE_ITER_A, batchsize=BATCH_SIZE, ctx=CTX)
critic = Critic(n_action=action_dim, n_features=state_dim, learning_rate=LR_C, t_replace_iter=REPLACE_ITER_C, batchsize=BATCH_SIZE, ctx=CTX)

M = Memory(capacity=MEMORY_CAPACITY, dims=2*state_dim + action_dim + 1)
var = 3

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        M.store_transition(s, a, r/10, s_)
        if M.pointer > MEMORY_CAPACITY:
            var *= 0.9995
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim:state_dim+action_dim]
            b_r = b_M[:, -state_dim-1:-state_dim]
            b_s_ = b_M[:, -state_dim:]

            a_ = actor.choose_action_(b_s_)
            a_grad = critic.learn(b_s, b_a, b_r, b_s_, a_)
            actor.learn(b_s, a_grad)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), ' Explore: %.2f' % var,)
            if ep_reward > -1000:
                RENDER = True
            break