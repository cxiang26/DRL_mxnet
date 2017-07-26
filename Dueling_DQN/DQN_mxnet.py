import mxnet as mx
import numpy as np

np.random.seed(1)
mx.random.seed(1)

class DuelingDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 replace_target_iter=200,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 ctx=mx.cpu(),
                 dueling=True,
                 param_file = None,
                 ):
        self.dueling = dueling
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.ctx = ctx
        self.cost_his = []
        self.count = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.Qnet = self.createQNetwork()
        self.target = self.createQNetwork(isTrain=False)

        if param_file!=None:
            self.Qnet.load_params(param_file)
        self.copyTargetQNetwork()

    def sym(self,predict=False):
        data = mx.sym.Variable('data')
        yInput = mx.sym.Variable('yInput')
        action = mx.sym.Variable('action')
        if self.dueling:
            f1 = mx.sym.FullyConnected(data=data, num_hidden=20, name='f1')
            relu1 = mx.sym.Activation(data=f1, act_type='relu', name='relu1')
            V = mx.sym.FullyConnected(data=relu1, num_hidden=1, name='V')
            A = mx.sym.FullyConnected(data=relu1, num_hidden=self.n_actions, name='A')
            Qvalue = mx.sym.broadcast_add(V, mx.sym.broadcast_sub(A, mx.sym.sum(A, axis=1, keepdims=True)/self.n_actions))
        else:
            f1 = mx.sym.FullyConnected(data=data,num_hidden=20,name='f1')
            relu1 = mx.sym.Activation(data=f1,act_type='relu',name='relu1')
            Qvalue = mx.sym.FullyConnected(data=relu1, num_hidden=self.n_actions,name='qvalue')
        output = mx.sym.sum((Qvalue*action - yInput)**2, axis=1)/self.n_actions
        loss=mx.sym.MakeLoss(output)
        a = mx.sym.Group([loss, mx.sym.BlockGrad(Qvalue)])
        #mx.viz.plot_network(a,shape = {"data":(self.batch_size,self.n_features)}).view(filename= 'DuelingDQN')
        if predict:
            return Qvalue
        else:
            return a

    def createQNetwork(self,bef_args=None,isTrain=True):
        if isTrain:
            modQ = mx.mod.Module(symbol=self.sym(), data_names=('data',), label_names=('action', 'yInput'), context=self.ctx)
            batch = self.batch_size
            modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        label_shapes=[('action',(batch,self.n_actions)), ('yInput',(batch,self.n_actions))],
                        for_training=isTrain)

            modQ.init_params(initializer=mx.init.Normal(0.3), arg_params=bef_args)
            modQ.init_optimizer(
                optimizer='RMSProp',
                optimizer_params={'learning_rate': self.lr})

        else:
            modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=self.ctx)
            batch = 1
            modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        for_training=isTrain)
            modQ.init_params(initializer=mx.init.Normal(0.3), arg_params=bef_args)

        return modQ

    def copyTargetQNetwork(self):
        arg_params,aux_params=self.Qnet.get_params()
        self.target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)
        self.count += 1
        print 'time to copy',self.count

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copyTargetQNetwork()
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        minibatch = self.memory[sample_index, :]
        state_batch = minibatch[:, :self.n_features]
        nextState_batch = minibatch[:, -self.n_features:]
        reward = minibatch[:, self.n_features + 1]

        q_next_s = []
        for i in range(self.batch_size):
            nextState_batch_temp = nextState_batch[i][np.newaxis,:]
            self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch_temp,self.ctx)],[]))
            q_next_s.append(self.target.get_outputs()[0].asnumpy()[0])
        q_target = np.zeros([self.batch_size, self.n_actions])
        act_one_hot = np.zeros([self.batch_size, self.n_actions])
        eval_act_index = minibatch[:, self.n_features].astype(int)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        act_one_hot[batch_index, eval_act_index] = 1.
        q_temp = np.squeeze(q_next_s)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_temp, axis=1)

        self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,self.ctx)],[mx.nd.array(act_one_hot, self.ctx), mx.nd.array(q_target, self.ctx)]),is_train=True)
        self.cost = np.sum(self.Qnet.get_outputs()[0].asnumpy())/self.batch_size
        self.Qnet.backward()
        self.Qnet.update()
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            self.target.forward(mx.io.DataBatch([mx.nd.array(observation,self.ctx)],[]), is_train=False)
            actions_value = self.target.get_outputs()[0].asnumpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
