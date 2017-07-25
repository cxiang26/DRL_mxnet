import mxnet as mx
import numpy as np

np.random.seed(1)
mx.random.seed(1)
class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(leaf_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        left_child_idx = 2*parent_idx + 1
        right_child_idx = left_child_idx + 1
        if left_child_idx >= len(self.tree):
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]

class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

class DQNPrioritizedReplay:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.005,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=500,
                 memory_size=10000,
                 batch_size=32,
                 ctx = mx.gpu(), #cpu() or gpu()
                 prioritized=True,
                 e_greedy_increment=None,
                 param_file = None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.ctx = ctx
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.cost_his = []
        self.count = 0
        self.prioritized = prioritized
        self.learn_step_counter = 0
        if self.prioritized:
            self.memory = Memory(capacity=self.memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.n_features*2+2))

        self.target = self.createQNetwork(isTrain=False) # building target_net
        self.Qnet = self.createQNetwork() # building eval_net
        if param_file!=None:
            self.Qnet.load_params(param_file)
        self.copyTargetQNetwork()

    def sym(self,predict=False):
        data = mx.sym.Variable('data')
        yInput = mx.sym.Variable('yInput')
        action = mx.sym.Variable('action')


        f1 = mx.sym.FullyConnected(data=data,num_hidden=20,name='f1')
        relu1 = mx.sym.Activation(data=f1,act_type='relu',name='relu1')
        Qvalue = mx.sym.FullyConnected(data=relu1, num_hidden=self.n_actions,name='qvalue')
        abs_errors = mx.sym.sum(mx.sym.abs(Qvalue*action - yInput), axis=1)
        output = mx.sym.sum((Qvalue * action - yInput)**2, axis=1)/self.n_actions
        loss = mx.sym.MakeLoss(output)
        a = mx.sym.Group([loss, mx.sym.BlockGrad(abs_errors)])

        if predict:
            return Qvalue
        else:
            return a

    def createQNetwork(self,bef_args=None,isTrain=True):
        if self.prioritized:
            if isTrain:
                modQ = mx.mod.Module(symbol=self.sym(), data_names=('data',), label_names=('yInput','action'), context=self.ctx)
                batch = self.batch_size
                modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        label_shapes=[('yInput',(batch,self.n_actions)),('action',(batch,self.n_actions))],
                        for_training=isTrain)

                modQ.init_params(initializer=mx.init.Normal(0.5), arg_params=bef_args)
                modQ.init_optimizer(
                    optimizer='RMSProp',
                    optimizer_params={
                        'learning_rate': self.lr
                })
            else:
                modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=self.ctx)
                batch = 1
                modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                            for_training=isTrain)

                modQ.init_params(initializer=mx.init.Normal(0.5), arg_params=bef_args)
        else:
            if isTrain:
                modQ = mx.mod.Module(symbol=self.sym(), data_names=('data',), label_names=('yInput','action'), context=self.ctx)
                batch = self.batch_size
                modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        label_shapes=[('yInput',(batch,self.n_actions)),('action',(batch,self.n_actions))],
                        for_training=isTrain)

                modQ.init_params(initializer=mx.init.Normal(0.5), arg_params=bef_args)
                modQ.init_optimizer(
                    optimizer='RMSProp',
                    optimizer_params={
                        'learning_rate': self.lr
                })
            else:
                modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=self.ctx)
                batch = 1
                modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                            for_training=isTrain)

                modQ.init_params(initializer=mx.init.Normal(0.5), arg_params=bef_args)

        return modQ

    def copyTargetQNetwork(self):
        arg_params,aux_params=self.Qnet.get_params()
        self.target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)
        self.count += 1
        print 'time to copy',self.count

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copyTargetQNetwork()

        if self.prioritized:
            tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
        else:
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            minibatch = self.memory[sample_index, :]
        state_batch = minibatch[:, :self.n_features]
        nextState_batch = minibatch[:, -self.n_features:]
        reward = minibatch[:, self.n_features + 1]

        q_next=[]
        for i in range(self.batch_size):
            nextState_batch_temp = nextState_batch[i][np.newaxis,:]
            self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch_temp,self.ctx)],[]))
            q_next.append(self.target.get_outputs()[0].asnumpy()[0])
        q_target = np.zeros((self.batch_size,self.n_actions))
        action_onehot = np.zeros((self.batch_size,self.n_actions))
        eval_act_index = minibatch[:, self.n_features].astype(int)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        action_onehot[batch_index, eval_act_index] = 1.
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,self.ctx)],[mx.nd.array(q_target, self.ctx), mx.nd.array(action_onehot, self.ctx)]),is_train=False)
            self.cost = np.sum(self.Qnet.get_outputs()[0].asnumpy())/self.batch_size
            abs_errors = self.Qnet.get_outputs()[1].asnumpy()
            for i in range(len(tree_idx)):
                idx = tree_idx[i]
                self.memory.update(idx, abs_errors[i])
            self.Qnet.backward()
            self.Qnet.update()
            self.cost_his.append(self.cost)

        else:
            self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,self.ctx)],[mx.nd.array(q_target, self.ctx), mx.nd.array(action_onehot, self.ctx)]),is_train=True)
            self.cost = np.sum(self.Qnet.get_outputs()[0].asnumpy())/self.batch_size
            self.Qnet.backward()
            self.Qnet.update()
            self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def store_transition(self, s, a, r, s_):
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        self.target.forward(mx.io.DataBatch([mx.nd.array(observation,self.ctx)],[]))
        if np.random.uniform() < self.epsilon:
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
