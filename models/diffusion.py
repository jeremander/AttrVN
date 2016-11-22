from sbm import *
from matplotlib.animation import FuncAnimation
from matplotlib import colors


class DiffusionGraph(TwoBlockSBMGraph):
    """TwoBlockSBMGraph subclass that can implement diffusion for vertex nomination."""
    def __init__(self, blocks_by_node, observed_flags = None, **kwargs):
        super().__init__(blocks_by_node, observed_flags)
        self.diffusion_init(**kwargs)  # use default diffusion initialization
    def diffusion_init(self, seed_temps = None, proportional = False):
        """Initializes temperatures for each node. Those whose membership is observed get values according to seed_temps (2-long array for 0 and 1, respectively). The remaining nodes get values equal to the average of the observed nodes. If seed_temps is None and proportional = False, default seed_temps are [-1, 1], otherwise default seed_temps are [-1 / (# observed 0), 1 / (# observed 1)] scaled so that the larger absolute value is 1."""
        num_seeds = self.observed_flags.sum()
        if (seed_temps is None):
            if proportional:
                self.seed_temps = np.array([-1.0 / (self.n - num_seeds), 1.0 / num_seeds])
                self.seed_temps /= np.abs(self.seed_temps).max()
            else:
                self.seed_temps = np.array([-1.0, 1.0])
        else:
            self.seed_temps = seed_temps
        num_pos_seeds = self.blocks_by_node[self.observed_flags].sum()
        num_neg_seeds = num_seeds - num_pos_seeds
        mean_temp = (num_neg_seeds * self.seed_temps[0] + num_pos_seeds * self.seed_temps[1]) / num_seeds
        self.temps = np.zeros(self.n, dtype = float)
        for (i, (block, flag)) in enumerate(zip(self.blocks_by_node, self.observed_flags)):
            self.temps[i] = self.seed_temps[block] if flag else mean_temp
    def diffusion_step(self, rate):
        if (not hasattr(self, 'L')):
            self.L = nx.laplacian_matrix(self)
        dt_temp = -self.L * self.temps * ~(self.observed_flags)
        self.temps += rate * dt_temp
        error = rate * np.linalg.norm(dt_temp)
        return error
    def diffusion(self, rate, max_iters = None, tol = 1e-8, verbosity = 0):
        counter = itertools.count() if (max_iters is None) else range(max_iters)
        for i in counter:
            error = self.diffusion_step(rate)
            if (verbosity > 1):
                print("Iteration %d (error %.3e)" % ((i + 1), error))
            if (error <= tol):
                if (verbosity > 0):
                    print("Terminated after %d iterations (error = %g)" % ((i + 1), error))
                return error
        return error
    def diffusion_anim(self, rate, max_iters = None, tol = 1e-8, fps = 20):
        plt.clf()
        fig = plt.gcf()
        pos = nx.spring_layout(self)
        nodes = nx.draw_networkx_nodes(self, pos = pos, node_color = np.array([self.get_color(self.temps[i]) for i in range(self.n)]), node_size = 100)
        edges = nx.draw_networkx_edges(self, pos = pos)
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        def frame_gen():
            counter = itertools.count() if (max_iters is None) else range(max_iters)
            for i in counter:
                error = self.diffusion_step(rate)
                if (error <= tol):
                    print("Terminated after %d iterations (error = %g)" % ((i + 1), error))
                    break
                else:
                    yield i
        def animate(n, nodes = nodes):
            nodes.set_array(np.array([self.get_color(self.temps[i]) for i in range(self.n)]))
            return nodes,
        #anim = FuncAnimation(fig, animate, frames = max_iters, blit = True)
        anim = FuncAnimation(fig, animate, frames = frame_gen(), blit = True)
        anim.save_count = 1000000  # max allowable frames
        anim.save('diffusion.mp4', fps = fps)
    def get_color(self, temp):
        x = (temp - self.seed_temps[0]) / (self.seed_temps[1] - self.seed_temps[0])
        return x
        #return 0.5 + (0.3 * (x - 0.5))
        #return (2 * max(0.0, x - 0.5), 0, 2 * max(0, 0.5 - x), 1.0)
    def draw(self, with_labels = False, seed = 123):
        """Colors the nodes red (high temp) or blue (low temp), interpolating colors appropriately."""
        if (seed is not None):
            np.random.seed(seed)
        plt.figure()
        nx.draw_networkx(self, node_color = [self.get_color(self.temps[i]) for i in range(self.n)], with_labels = with_labels, node_size = 100) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.show(block = False)
    @classmethod
    def from_vngraph(cls, graph, *args, **kwargs):
        """Convenience constructor for initalizing DiffusionGraph from a VNGraph containing attributes 'blocks_by_node' and 'observed_flags'."""
        g = cls(graph.blocks_by_node, graph.observed_flags, *args, **kwargs)
        g.copy_edges_from_graph(graph)
        return g



p00 = 0.1
p01 = 0.01
p11 = 0.1
p1 = 0.2
n = 200


g = TwoBlockSBM(p00, p01, p11, p1).sample(n)
g2 = DiffusionGraph.from_vngraph(g.mcar_occlude(int(0.75 * n)))


