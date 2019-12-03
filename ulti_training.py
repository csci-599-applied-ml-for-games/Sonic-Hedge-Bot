import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

resume = True
restore_file = "neat-checkpoint-2642"


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def show_input(self,ob, iny, inx):
    scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    scaledimg = cv2.resize(scaledimg, (iny, inx))
    cv2.imshow('main', scaledimg)
    cv2.waitKey(1)


    def load_picke(self,filename)
        parser = argparse.ArgumentParser(description='execute the model')
        parser.add_argument('-input, -i', dest='input', action='store',
                    required=True,help='Winner structure')
        args = parser.parse_args()


        with open(args.input, 'rb') as file:
        winner = pkl.load(file)

        print (winner)

        print('\nOutput:')
    def work(self):

        self.env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1")

        self.env.reset()
        #for genome_id, genome in self.genomes:
        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False
        #net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        #fitness_current = 0
        #xpos = 0
        #frame=0
        xpos_max = 0
        counter = 0
        imgarray = []
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        
        done = False

        while not done:
            #self.env.render()
            #frame+=1
            # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)

            xpos = info['x']
            
            if xpos >= 65664:
                    fitness_current += 10000000
                    done = True
            
            fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 250:
                done = True
                print( fitness_current)
                
            self.genome.fitness = fitness_current

            #xpos = info['x']

            '''if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 10
            else:
                counter += 1

            if counter > 250:
                done = True

        print(fitness)'''
        return fitness_current


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