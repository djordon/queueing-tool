from .queues_agents import Agent


class LearningAgent(Agent) :

    def __init__(self, issn) :
        Agent.__init__(self, issn)

    def __repr__(self) :
        return "LearningAgent. issn: %s, time: %s" % (self.issn, self.time)


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def desired_destination(self, *info) :
        return self.dest


    def set_dest(self, net=None, dest=None) :
        self.dest = int(dest)

