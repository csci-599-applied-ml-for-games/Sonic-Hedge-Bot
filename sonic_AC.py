import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

resume = True
restore_file = "neat-checkpoint-517"

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        ....
        ....
        ....
        if scope != 'global':
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            #Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            #Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss,local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

            #Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        self.env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1")

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        frame=0
        xpos_max = 0
        counter = 0
        imgarray = []

        while not done:
            self.env.render()
            frame+=1
            # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)

            xpos = info['x']

            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 10
            else:
                counter += 1

            if counter > 250:
                done = True

        print(fitness)
        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

with tf.device("/cpu:0"): 
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,saver,model_path))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,master_network,sess,coord)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)