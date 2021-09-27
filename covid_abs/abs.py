"""
Main code for Agent Based Simulation
"""

from covid_abs.agents import Status, InfectionSeverity, Agent
from covid_abs.common import *
import numpy as np

def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Simulation(object):
    def __init__(self, **kwargs):
        self.population = [] # population of agents
        self.population_size = kwargs.get("population_size", 300) # number of people
        self.length = kwargs.get("length", 162) # length of environment
        self.height = kwargs.get("height", 162) # height of environment
        self.initial_infected_perc = kwargs.get("initial_infected_perc", 0.01) # initial percent of infected people
        self.initial_immune_perc = kwargs.get("initial_immune_perc", 0.01) # initial pecent of immune people
        self.contagion_distance = kwargs.get("contagion_distance", 1.) # minimal contact distance
        self.contagion_rate = kwargs.get("contagion_rate", 0.9) # probability of contagion in case of contact
        self.symptomatic_rate = kwargs.get("contagion_rate", 0.75) # percentage of infected people that has symptoms
        self.critical_limit = kwargs.get("critical_limit", 0.01) # percent of population which the Health System can afford
        self.amplitudes = kwargs.get('amplitudes',
                                     {Status.Susceptible: 10,
                                      Status.Recovered_Immune: 10,
                                      Status.Infected: 10})
        # A dictionary with the average mobility of agents inside the shared environment for each status
        self.minimum_income = kwargs.get("minimum_income", 1180) # base (or minimum) daily income, related to the most poor wealth quintile
        self.minimum_expense = kwargs.get("minimum_expense", 764) # The base (or minimum) daily expense, related to the most poor wealth quintile
        self.statistics = None # dictionary with the population statistics for the current iteration
        self.triggers_simulation = kwargs.get("triggers_simulation", []) # dictionary with conditional changes in the Simulation attributes
        self.triggers_population = kwargs.get("triggers_population", []) # dictionary with conditional changes in the Agent attributes
        self.mode = kwargs.get('mode', 'normal')# The mode of countermeasures: 'masks', 'tests', 'maskstests', 'lockdown' "
        self.original_contagion_rate = kwargs.get("original_contagion_rate", 0.9) # original contagion rate
        self.original_contagion_distance = kwargs.get('original_contagion_distance', 1) # original contagion distance
        self.mask_rate = kwargs.get('mask_rate', 0.3) # contagion rate when masks are used
        self.mask_distance = kwargs.get('mask_distance', 0.5) # contagion distance when masks are used
        self.tests = kwargs.get('tests', False) # tests bool
        self.test_rate = kwargs.get("test_rate", 0.03) # rate of people that are randomly tested if tests=True
        self.test_accuracy = kwargs.get("test_accuracy", 0.95) # accuracy of tests
        self.lockdown = kwargs.get('lockdown', False) # lockdown bool
        self.policy = kwargs.get('policy', 'normal') # policy info
        self.vaccination = kwargs.get('vaccination', False) # vaccination bool
        self.vac_rate = kwargs.get('vac_rate', 0.01) # percentage of whole population (including immune) that gets vaccinated per day
        self.contact_tracing = kwargs.get('contact_tracing', False) # contact tracing bool
        self.traced_contacts = [] # list of traced contacts, old contacts deleted every day
        self.lim_mobility = kwargs.get('lim_mobility', False) #limited mobility
        
        self.total_wealth = kwargs.get("total_wealth", 10**7)

    def _xclip(self, x):
        return np.clip(int(x), 0, self.length)

    def _yclip(self, y):
        return np.clip(int(y), 0, self.height)

    def get_population(self):
        """
        Return the population in the current iteration
        """
        return self.population

    def set_population(self, pop):
        """
        Update the population in the current iteration
        """
        self.population = pop

    def set_amplitudes(self, amp):
        self.amplitudes = amp

    def append_trigger_simulation(self, condition, attribute, action):
        """
        Append a conditional change in the Simulation attributes

        :param condition: a lambda function that receives the current simulation instance and
        returns a boolean
        :param attribute: string, the attribute name of the Simulation which will be changed
        :param action: a lambda function that receives the current simulation instance and returns
        the new value of the attribute
        """
        self.triggers_simulation.append({'condition': condition, 'attribute': attribute, 'action': action})

    def append_trigger_population(self, condition, attribute, action):
        """
        Append a conditional change in the population attributes

        :param condition: a lambda function that receives the current agent instance and returns a boolean
        :param attribute: string, the attribute name of the agent which will be changed
        :param action: a lambda function that receives the current agent instance and returns the new
        value of the attribute
        """
        self.triggers_population.append({'condition': condition, 'attribute': attribute, 'action': action})

    def random_position(self):
        x = self._xclip(self.length / 2 + (np.random.randn(1) * (self.length / 3)))
        y = self._yclip(self.height / 2 + (np.random.randn(1) * (self.height / 3)))

        return x, y

    def create_agent(self, status):
        """
        Create a new agent with the given status

        :param status: a value of agents.Status enum
        :return: the newly created agent
        """
        x, y = self.random_position()

        age = int(np.random.beta(2, 5, 1) * 100)
        social_stratum = int(np.random.rand(1) * 100 // 20)
        self.population.append(Agent(x=x, y=y, age=age, status=status, social_stratum=social_stratum))

    def initialize(self):
        """
        Initializate the Simulation by creating its population of agents
        """
        # initialize mode
        self.original_contagion_rate = self.contagion_rate 
        
        if self.mode == 'tests':
            self.tests = True
            
        if self.mode == 'masks':
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            
        if self.mode == 'maskstests':
            self.tests = True
            self.contagion_rate = self.mask_rate
            
        if self.mode == 'lockdown':
            self.lockdown = True
            
        if self.mode == 'contact_tracing':
            self.contact_tracing = True
            
        if self.mode == 'stage1':
           self.contagion_rate = self.mask_rate
           self.contagion_distance = self.mask_distance
            
        if self.mode == 'stage2':
            self.contagion_rate = self.mask_rate
            self.tests = True
            self.contagion_distance = self.mask_distance
            
        if self.mode == 'stage3':
            self.contagion_rate = self.mask_rate
            self.tests = True
            self.contact_tracing = True
            self.contagion_distance = self.mask_distance
            
        if self.mode == 'stage4':
            self.contagion_rate = self.mask_rate
            self.tests = True
            self.contact_tracing = True
            self.lim_mobility = True
            self.contagion_distance = self.mask_distance
            
        if self.mode == 'stage5':
            self.lockdown = True
        ############################################

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible)

        # Share the common wealth of 10^4 among the population, according each agent social stratum
        for quintile in [0, 1, 2, 3, 4]:
            total = lorenz_curve[quintile] * self.total_wealth
            qty = max(1.0, np.sum([1 for a in self.population if a.social_stratum == quintile and a.age >= 18]))
            ag_share = total / qty
            for agent in filter(lambda x: x.social_stratum == quintile and x.age >= 18, self.population):
                agent.wealth = ag_share

    def contact(self, agent1, agent2):
        """
        Performs the actions needed when two agents get in touch.

        :param agent1: an instance of agents.Agent
        :param agent2: an instance of agents.Agent
        """

        if agent1.status == Status.Susceptible and agent2.status == Status.Infected:
            contagion_test = np.random.random()
            #agent1.infection_status = InfectionSeverity.Exposed
            if contagion_test <= self.contagion_rate:
                agent1.status = Status.Infected
                agent1.infected_status = InfectionSeverity.Asymptomatic
                '''symptomatic_test = np.random.random()
                if symptomatic_test <= self.symptomatic_rate:
                    agent1.infected_status = InfectionSeverity.Symptomatic
                    agent1.isolated = True
                else:
                    agent1.infected_status = InfectionSeverity.Asymptomatic'''

    def move(self, agent, triggers=[]):
        """
        Performs the actions related with the movement of the agents in the shared environment

        :param agent: an instance of agents.Agent
        :param triggers: the list of population triggers related to the movement
        """
        
        if agent.status == Status.Death or (agent.status == Status.Infected
                                            and (agent.infected_status == InfectionSeverity.Severe)):
            return

        for trigger in triggers:
            if trigger['condition'](agent):
                agent.x, agent.y = trigger['action'](agent)
                return

        ix = int(np.random.randn(1) * self.amplitudes[agent.status])
        iy = int(np.random.randn(1) * self.amplitudes[agent.status])

        if (agent.x + ix) <= 0 or (agent.x + ix) >= self.length:
            agent.x -= ix
        else:
            agent.x += ix

        if (agent.y + iy) <= 0 or (agent.y + iy) >= self.height:
            agent.y -= iy
        else:
            agent.y += iy

        dist = np.sqrt(ix ** 2 + iy ** 2)
        result_ecom = np.random.rand(1)
        agent.wealth += dist * result_ecom * self.minimum_expense * basic_income[agent.social_stratum]

    def update(self, agent):
        """
        Update the status of the agent
        """

        if agent.status == Status.Death:
            return

        if agent.status == Status.Infected:
            agent.infected_time += 1

            indice = agent.age // 10 - 1 if agent.age > 10 else 0
            
            if agent.infected_time == 1:
                if agent.infected_status == InfectionSeverity.Asymptomatic:
                    symptomatic_test = np.random.random()
                    if symptomatic_test <= self.symptomatic_rate:
                        agent.infected_status = InfectionSeverity.Symptomatic
                        agent.isolated = True
                        agent.protection_known = True

            teste_sub = np.random.random()

            if agent.infected_status == InfectionSeverity.Symptomatic:
                if age_severe_probs[indice] > teste_sub:
                    agent.infected_status = InfectionSeverity.Severe
                    agent.isolated = False
            if agent.infected_status == InfectionSeverity.Severe:
                self.get_statistics()
                if self.statistics['Severe']  >= self.critical_limit:
                    agent.status = Status.Death
                    agent.infected_status = InfectionSeverity.Asymptomatic
                self.statistics = None
                death_test = np.random.random()
                if age_death_probs[indice] > death_test:
                    agent.status = Status.Death
                    agent.infected_status = InfectionSeverity.Asymptomatic
                    return

            if agent.infected_time > 20:
                agent.infected_time = 0
                agent.status = Status.Recovered_Immune
                agent.infected_status = InfectionSeverity.Asymptomatic
                agent.isolated = False
                
        # Random Tests
        if self.tests == True:
            test_test = np.random.random()
            if test_test < self.test_rate:
                covid_test = np.random.random()
                if covid_test <= self.test_accuracy: 
                    if agent.infected_status == InfectionSeverity.Asymptomatic and agent.status == Status.Infected:
                        agent.isolated = True
                        agent.protection_known = True
                    if agent.status == Status.Susceptible:
                        agent.isolated = False
                else:
                    if agent.status == Status.Susceptible:
                        agent.isolated = True
                        agent.protection_known = True
        
        # Vaccination
        if self.vaccination == True:
            vac_test =  np.random.random()
            if vac_test < self.vac_rate:
                if agent.status == Status.Susceptible or (agent.status == Status.Infected and agent.infected_status == InfectionSeverity.Asymptomatic):
                    agent.status = Status.Recovered_Immune
                    agent.protection_known = True


        agent.wealth -= self.minimum_expense * basic_income[agent.social_stratum]

    def execute(self):
        """
        Execute a complete iteration cycle of the Simulation, executing all actions for each agent
        in the population and updating the statistics
        """
        mov_triggers = [k for k in self.triggers_population if k['attribute'] == 'move']
        other_triggers = [k for k in self.triggers_population if k['attribute'] != 'move']

        for agent in self.population:
            self.move(agent, triggers=mov_triggers)
            self.update(agent)

            for trigger in other_triggers:
                if trigger['condition'](agent):
                    attr = trigger['attribute']
                    agent.__dict__[attr] = trigger['action'](agent.__dict__[attr])

        dist = np.zeros((self.population_size, self.population_size))

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]

                if distance(ai, aj) <= self.contagion_distance:
                    contacts.append((i, j))

        for par in contacts:
            ai = self.population[par[0]]
            aj = self.population[par[1]]
            self.contact(ai, aj)
            self.contact(aj, ai)

        if len(self.triggers_simulation) > 0:
            for trigger in self.triggers_simulation:
                if trigger['condition'](self):
                    attr = trigger['attribute']
                    self.__dict__[attr] = trigger['action'](self.__dict__[attr])

        self.statistics = None

    def get_positions(self):
        """Return the list of x,y positions for all agents"""
        return [[a.x, a.y] for a in self.population]

    def get_description(self, complete=False):
        """
        Return the list of Status and InfectionSeverity for all agents

        :param complete: a flag indicating if the list must contain the InfectionSeverity (complete=True)
        """
        if complete:
            return [a.get_description() for a in self.population]
        else:
            return [a.status.name for a in self.population]

    def get_statistics(self, kind='info'):
        """
        Calculate and return the dictionary of the population statistics for the current iteration.

        :param kind: 'info' for health statiscs, 'ecom' for economic statistics and None for all statistics
        :return: a dictionary
        """
        if self.statistics is None:
            self.statistics = {}
            for status in Status:
                self.statistics[status.name] = np.sum(
                    [1 for a in self.population if a.status == status]) / self.population_size

            for infected_status in filter(lambda x: x != InfectionSeverity.Exposed, InfectionSeverity):
                self.statistics[infected_status.name] = np.sum([1 for a in self.population if
                                                                a.infected_status == infected_status and
                                                                a.status != Status.Death]) / self.population_size

            for quintile in [0, 1, 2, 3, 4]:
                self.statistics['Q{}'.format(quintile + 1)] = np.sum(
                    [a.wealth for a in self.population if a.social_stratum == quintile
                     and a.age >= 18 and a.status != Status.Death])

        return self.filter_stats(kind)

    def filter_stats(self, kind):
        if kind == 'info':
            return {k: v for k, v in self.statistics.items() if not k.startswith('Q') and k not in ('Business','Government')}
        elif kind == 'ecom':
            return {k: v for k, v in self.statistics.items() if k.startswith('Q') or k in ('Business','Government')}
        else:
            return self.statistics

    def __str__(self):
        return str(self.get_description())


class MultiPopulationSimulation(Simulation):
    def __init__(self, **kwargs):
        super(MultiPopulationSimulation, self).__init__(**kwargs)
        self.simulations = kwargs.get('simulations', [])
        self.positions = kwargs.get('positions', [])
        self.total_population = kwargs.get('total_population', 0)

    def get_population(self):
        population = []
        for simulation in self.simulations:
            population.extend(simulation.get_population())
        return population

    def append(self, simulation, position):
        self.simulations.append(simulation)
        self.positions.append(position)
        self.total_population += simulation.population_size

    def initialize(self):
        for simulation in self.simulations:
            simulation.initialize()

    def execute(self, **kwargs):
        for simulation in self.simulations:
            simulation.execute()

        for m in np.arange(0, len(self.simulations)):
            for n in np.arange(m + 1, len(self.simulations)):

                for i in np.arange(0, self.simulations[m].population_size):
                    ai = self.simulations[m].get_population()[i]

                    for j in np.arange(0, self.simulations[n].population_size):
                        aj = self.simulations[n].get_population()[j]

                        if np.sqrt(((ai.x + self.positions[m][0]) - (aj.x + self.positions[n][0])) ** 2 +
                                   ((ai.y + self.positions[m][1]) - (
                                           aj.y + self.positions[n][1])) ** 2) <= self.contagion_distance:
                            self.simulations[m].contact(ai, aj)
                            self.simulations[n].contact(aj, ai)
        self.statistics = None

    def get_positions(self):
        positions = []
        for ct, simulation in enumerate(self.simulations):
            for a in simulation.get_population():
                positions.append([a.x + self.positions[ct][0], a.y + self.positions[ct][1]])
        return positions

    def get_description(self, complete=False):
        situacoes = []
        for simulation in self.simulations:
            for a in simulation.get_population():
                if complete:
                    situacoes.append(a.get_description())
                else:
                    situacoes.append(a.status.name)

        return situacoes

    def get_statistics(self, kind='info'):
        if self.statistics is None:

            self.statistics = {}
            for status in Status:
                for simulation in self.simulations:
                    self.statistics[status.name] = np.sum(
                        [1 for a in filter(lambda x: x.status == status, simulation.get_population())])
                self.statistics[status.name] /= self.total_population

            for infected_status in InfectionSeverity:
                for simulation in self.simulations:
                    self.statistics[infected_status.name] = np.sum(
                        [1 for a in filter(lambda x: x.infected_status == infected_status and x.status != Status.Death,
                                           simulation.get_population())])
                self.statistics[infected_status.name] /= self.total_population

            for quintil in [0, 1, 2, 3, 4]:
                for simulation in self.simulations:
                    key = 'Q{}'.format(quintil + 1)
                    self.statistics[key] = np.sum([a.wealth for a in simulation.get_population()
                                                   if a.social_stratum == quintil and a.age >= 18
                                                   and a.status != Status.Death])

        return self.filter_stats(kind)

    def __str__(self):
        return str(self.get_description())
