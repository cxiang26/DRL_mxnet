import mxnet as mx
import numpy as np

ctx=mx.cpu()

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 param_file = None,
                 ):
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
        self.cost_his = []
        self.count = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.target = self.createQNetwork(isTrain=False)
        self.Qnet = self.createQNetwork()
        if param_file!=None:
            self.Qnet.load_params(param_file)
        self.copyTargetQNetwork()
        # saving and loading networks

    def sym(self,predict=False):
        data = mx.sym.Variable('data')
        yInput = mx.sym.Variable('yInput')
        f1 = mx.sym.FullyConnected(data=data,num_hidden=10,name='f1')
        relu1 = mx.sym.Activation(data=f1,act_type='relu',name='relu1')
        Qvalue = mx.sym.FullyConnected(data=relu1, num_hidden=self.n_actions,name='qvalue')
        #temp = Qvalue
        #coeff = mx.sym.sum(temp,axis=1,name='temp1')
        output = mx.sym.sum((Qvalue - yInput)**2)/self.n_actions
        loss=mx.sym.MakeLoss(output)

        if predict:
            return Qvalue
        else:
            return loss

    def createQNetwork(self,bef_args=None,isTrain=True):
        if isTrain:
            modQ = mx.mod.Module(symbol=self.sym(), data_names=('data',), label_names=('yInput',), context=ctx)
            batch = self.batch_size
            modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        label_shapes=[('yInput',(batch,self.n_actions))],
                        for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)
            modQ.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': 0.0002,
                    'wd': 0.,
                    'beta1': 0.5,
            })
        else:
            modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=ctx)
            batch = 1
            modQ.bind(data_shapes=[('data',(batch,self.n_features))],
                        for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)

        return modQ

    def copyTargetQNetwork(self):
        arg_params,aux_params=self.Qnet.get_params()
        #arg={}
        #for k,v in arg_params.iteritems():
        #    arg[k]=arg_params[k].asnumpy()

        self.target.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)

        #args,auxs=self.target.get_params()
        #arg1={}
        #for k,v in args.iteritems():
        #    arg1[k]=args[k].asnumpy()
        self.count += 1
        print 'time to copy',self.count

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copyTargetQNetwork()
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # Step 1: obtain random minibatch from replay memory
        minibatch = self.memory[sample_index, :]
        state_batch = minibatch[:, :self.n_features]
        nextState_batch = minibatch[:, -self.n_features:]
        reward = minibatch[:, self.n_features + 1]
        # Step 2: calculate y
        q_next=[]
        q_target=[]
        for i in range(self.batch_size):
            state_batch_temp = state_batch[i][np.newaxis,:]
            nextState_batch_temp = nextState_batch[i][np.newaxis,:]
            self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch_temp,ctx)],[]))
            q_next.append(self.target.get_outputs()[0].asnumpy()[0])
            self.target.forward(mx.io.DataBatch([mx.nd.array(state_batch_temp,ctx)],[]))
            q_target.append(self.target.get_outputs()[0].asnumpy()[0])
        q_target = np.squeeze(q_target)
        eval_act_index = minibatch[:, self.n_features].astype(int)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,ctx)],[mx.nd.array(q_target, ctx)]),is_train=True)
        self.cost = self.Qnet.get_outputs()[0].asnumpy()
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
        self.target.forward(mx.io.DataBatch([mx.nd.array(observation,ctx)],[]))
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
