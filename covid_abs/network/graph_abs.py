"""
Graph induced
"""

import numpy as np
from covid_abs.abs import *
from covid_abs.agents import *
from covid_abs.network.agents import EconomicalStatus, Business, House, Person
from covid_abs.network.util import new_day, work_day, new_month, bed_time, work_time, lunch_time, free_time, simulation_day


class GraphSimulation(Simulation):
    def __init__(self, **kwargs):
        super(GraphSimulation, self).__init__(**kwargs)
        self.total_population = kwargs.get('total_population', 0)
        self.total_business = kwargs.get('total_business', 10) # number of businesses
        self.business_distance = kwargs.get('business_distance', 20) # distance between businesses
        self.government = None # the givernment agent
        self.business = [] # list of businesses
        self.houses = [] # list of houses
        self.healthcare = None # the healthcare agent
        self.homeless_rate = kwargs.get("homeless_rate", 0.00) # homeless rate 
        self.unemployment_rate = kwargs.get("unemployment_rate", 0.056) # unemployment rate
        self.homemates_avg = kwargs.get("homemates_avg", 3) # average number of people per house
        self.homemates_std = kwargs.get("homemates_std", 1) # std of number of people per house
        self.iteration = -1 # current iteration
        self.callbacks = kwargs.get('callbacks', {}) # callbacks (functions for routines)
        self.public_gdp_share = kwargs.get('public_gdp_share', 0.1) # public gdp share
        self.business_gdp_share = kwargs.get('business_gdp_share', 0.5) # business gdp share
        self.incubation_time = kwargs.get('incubation_time', 5) # end of incubation time 
        self.contagion_time = kwargs.get('contagion_time', 10) # end of contagion time
        self.recovering_time = kwargs.get('recovering_time', 20) # end of recovery time
        self.reward = 0 # reward for RL
        self.observation = np.zeros(5, dtype=float) # observation for RL = input for the NN

    def register_callback(self, event, action):
        self.callbacks[event] = action

    def callback(self, event, *args):
        if event in self.callbacks:
            return self.callbacks[event](*args)

        return False

    def get_unemployed(self):
        return [p for p in self.population if p.is_unemployed()
                and p.status != Status.Death and p.infected_status == InfectionSeverity.Asymptomatic]

    def get_homeless(self):
        return [p for p in self.population if p.is_homeless()
                and p.status != Status.Death and p.infected_status == InfectionSeverity.Asymptomatic]

    def create_business(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.business.append(Business(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                      #fixed_expenses=(social_stratum+1)*self.minimum_expense
                                      #fixed_expenses=self.minimum_expense / (5 - social_stratum)
                                      environment=self
                                      ))

    def create_house(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        house = House(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                 #fixed_expenses=(social_stratum+1)*self.minimum_expense/(self.homemates_avg*10
                      environment=self)
        self.callback('on_create_house', house)
        self.houses.append(house)

    def create_agent(self, status, social_stratum=None, infected_time=0):
        """
        Create a new agent with the given status

        :param infected_time:
        :param social_stratum:
        :param status: a value of agents.Status enum
        :return: the newly created agent
        """

        age = int(np.random.beta(2, 5, 1) * 100)
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        person = Person(age=age, status=status, social_stratum=social_stratum, infected_time=infected_time,
                        environment=self)
        self.callback('on_create_person', person)
        self.population.append(person)
        

    def initialize(self):
        """
        Initialize the Simulation by creating its population of agents
        """

        self.callback('on_initialize', self)
        
        self.original_contagion_rate = self.contagion_rate 
        self.original_contagion_distance = self.contagion_distance
        
        
        # initialize the mode and policy
        ##################################
        if self.policy == 'simmplepolicy1':
            self.mode = 'masks'
            
        if self.policy == 'increasing':
            self.mode = 'normal'
        
        if self.policy == 'decreasing':
            self.mode = 'lockdown'
            
        if self.policy == 'hospcap':
            self.mode = 'masks'
        
        ###################################
        if self.mode == 'tests':
            self.tests = True
            
        if self.mode == 'masks':
            self.contagion_rate = self.mask_rate
            
        if self.mode == 'maskstests':
            self.tests = True
            self.contagion_rate = self.mask_rate
            
        if self.mode == 'lim_mobility':
            self.lim_mobility = True
            
        if self.mode == 'lockdown':
            self.lockdown = True
        
        if self.mode == 'contact_tracing':
            self.contact_tracing = True
            
        if self.mode == 'stage1':
           self.contagion_rate = self.mask_rate
            
        if self.mode == 'stage2':
            self.contagion_rate = self.mask_rate
            self.tests = True
            
        if self.mode == 'stage3':
            self.contagion_rate = self.mask_rate
            self.tests = True
            self.contact_tracing = True
            
        if self.mode == 'stage4':
            self.contagion_rate = self.mask_rate
            self.tests = True
            self.contact_tracing = True
            self.lim_mobility = True
            
        if self.mode == 'stage5':
            self.lockdown = True
            
        ##################################
        
        x, y = self.random_position()
        self.healthcare = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Healthcare, environment=self)
        self.healthcare.fixed_expenses += self.minimum_expense * 3
        x, y = self.random_position()
        self.government = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Government,
                                   social_stratum=4, price=1.0, environment=self)
        self.government.fixed_expenses += self.population_size * (self.minimum_expense*0.05)

        #number of houses
        for i in np.arange(0, int(self.population_size // self.homemates_avg)):
            self.create_house(social_stratum=i % 5)

        # number of business
        for i in np.arange(0, self.total_business):
            self.create_business(social_stratum=5 - (i % 5))

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected, infected_time=5)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible, social_stratum=i % 5)

        # Share the common wealth of 10^4 among the population, according each agent social stratum

        self.government.wealth = self.total_wealth * self.public_gdp_share

        for quintile in [0, 1, 2, 3, 4]:

            _houses = [x for x in filter(lambda x: x.social_stratum == quintile, self.houses)]
            nhouses = len(_houses)

            if nhouses == 0:
                self.create_house(social_stratum=quintile)
                _houses = [self.houses[-1]]
                nhouses = 1

            if self.total_business > 5:
                btotal = lorenz_curve[quintile] * (self.total_wealth * self.business_gdp_share)
                bqty = max(1.0, np.sum([1.0 for a in self.business if a.social_stratum == quintile]))
            else:
                btotal = self.total_wealth * self.business_gdp_share
                bqty = self.total_business

            ag_share = btotal / bqty
            for bus in filter(lambda x: x.social_stratum == quintile, self.business):
                bus.wealth = ag_share

            ptotal = lorenz_curve[quintile] * self.total_wealth * (1 - (self.public_gdp_share + self.business_gdp_share))

            pqty = max(1.0, np.sum([1 for a in self.population if
                                   a.social_stratum == quintile and a.economical_status == EconomicalStatus.Active]))
            ag_share = ptotal / pqty

            for agent in filter(lambda x: x.social_stratum == quintile, self.population):

                # distribute wealth

                if agent.economical_status == EconomicalStatus.Active:
                    agent.wealth = ag_share
                    agent.incomes = basic_income[agent.social_stratum] * self.minimum_income

                    # distribute employ

                    unemployed_test = np.random.rand()

                    if unemployed_test >= self.unemployment_rate:
                        ix = np.random.randint(0, self.total_business)
                        self.business[ix].hire(agent)

                agent.expenses = basic_income[agent.social_stratum] * self.minimum_expense

                #distribute habitation

                homeless_test = np.random.rand()

                if not (quintile == 0 and homeless_test <= self.homeless_rate):
                    for kp in range(0, 5):
                        ix = np.random.randint(0, nhouses)
                        house = _houses[ix]
                        if house.size < self.homemates_avg + self.homemates_std:
                            house.append_mate(agent)
                            continue
                    if agent.house is None:
                        ix = np.random.randint(0, nhouses)
                        self.houses[ix].append_mate(agent)

        self.callback('post_initialize', self)
       
        ##########################################
        # update observation and reward
        action = 0
        if self.mode == 'stage1':
            action = 0
        if self.mode == 'stage2':
            action = 1
        if self.mode == 'stage3':
            action = 2
        if self.mode == 'stage4':
            action = 3
        if self.mode == 'stage5':
            action = 4
            
        # compute reward of 10th day
        sev = np.sum([1 for a in self.population if
                      a.infected_status == InfectionSeverity.Severe])/self.population_size
        a = 0.4
        b= -0.1
        p = 1.5
        self.reward = a*((self.critical_limit-sev)/self.critical_limit)+b*(action/4)**p  
        
        
        
        self.statistics = None
        self.get_statistics(kind='info')
        incidence = 0
        incidence = np.sum([1 for a in self.population if
                              a.isolated == True or
                              a.infected_status == InfectionSeverity.Severe])
        quarantined = np.sum([1 for a in self.population if
                                  a.isolated == True])/self.population_size*100
        incidence = incidence/self.population_size *100
        known_immunity = np.sum([1 for a in self.population if 
                               a.status == Status.Recovered_Immune and
                               a.protection_known == True])/self.population_size*100
        self.observation[0] = incidence
        self.observation[1] = self.statistics['Severe']*100
        self.observation[2] = quarantined
        self.observation[3] = self.statistics['Death']*100
        self.observation[4] = known_immunity
        

    def execute(self):

        self.iteration += 1

        if self.callback('on_execute', self):
            return
        
        # sleep at night
        if not new_day(self.iteration) and bed_time(self.iteration):
            return 


        bed = bed_time(self.iteration)
        work = work_time(self.iteration)
        free = free_time(self.iteration)
        lunch = lunch_time(self.iteration)
        new_dy = new_day(self.iteration)
        work_dy = work_day(self.iteration)
        new_mth = new_month(self.iteration)
        day = simulation_day(self.iteration)

        #if new_dy:
        #    print("Day {}".format(self.iteration // 24))

        for agent in filter(lambda x: x.status != Status.Death, self.population):
            # quarantine all isolated people
            if agent.isolated == True:
                agent.move_to_home()
                moving = True
            else:
                moving = False
            
            #if not self.callback('on_person_move', agent):
            if self.lockdown == True:
                if agent.house is not None:
                    agent.house.checkin(agent)
                
            if not self.lockdown:
                if not moving:
                    if bed:
                        agent.move_to_home()

                    elif lunch or free or not work_dy:
                        if not self.lim_mobility == True:
                            agent.move_freely()
                        else:
                            agent.move_to_home()

                    elif work_dy and work:
                        agent.move_to_work()
    
                    
                    

            self.callback('post_person_move', agent)


            if new_dy:
                agent.update()
                
                # trace contacts if desired
                if self.contact_tracing == True:
                    self.test_contacts()
                    self.traced_contacts = []
                    
                    
                modus = self.mode
                changed = True
                
                # change mode according to policy
                self.check_policy()
                
                # check if the mode has changed
                if modus == self.mode:
                    changed = False
                
                # change mode variables only if the mode has changed
                if changed:
                    self.check_mode()
            
                    

            if agent.infected_status == InfectionSeverity.Asymptomatic:
                for bus in filter(lambda x: x != agent.employer, self.business):
                    if distance(agent, bus) <= self.business_distance:
                        bus.supply(agent)

        for bus in filter(lambda b: b.open, self.business):
            if new_dy:
                bus.update()

            if self.iteration > 1 and new_mth:
                bus.accounting()

        for house in filter(lambda h: h.size > 0, self.houses):
            if new_dy:
                house.update()

            if self.iteration > 1 and new_mth:
                house.accounting()

        if new_dy:
            self.government.update()
            self.healthcare.update()

        if self.iteration > 1 and new_mth:
            self.government.accounting()

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]
                if ai.status == Status.Death or ai.status == Status.Death:
                    continue

                if distance(ai, aj) <= self.contagion_distance:
                    contacts.append((i, j))
                    self.traced_contacts.append((i,j))

        for pair in contacts:
            ai = self.population[pair[0]]
            aj = self.population[pair[1]]
            self.contact(ai, aj)
            self.contact(aj, ai)

        self.statistics = None

        self.callback('post_execute', self)
         
        
        # save observations 
        self.statistics = None
        self.get_statistics(kind='info')
        incidence = 0         
        incidence = np.sum([1 for a in self.population if
                            a.isolated == True or
                            a.infected_status == InfectionSeverity.Severe])
        quarantined = np.sum([1 for a in self.population if
                                a.isolated == True])/self.population_size*100
        incidence = incidence/self.population_size *100
        self.observation[0] = incidence
        self.observation[1] = self.statistics['Severe']*100
        self.observation[2] = quarantined
        self.observation[3] = self.statistics['Death']*100
            
        self.observation = self.observation.astype(float)

    def contact(self, agent1, agent2):
        """
        Performs the actions needed when two agents get in touch.

        :param agent1: an instance of agents.Agent
        :param agent2: an instance of agents.Agent
        """

        if self.callback('on_contact', agent1, agent2):
            return

        if agent1.status == Status.Susceptible and agent2.status == Status.Infected:
            low = np.random.randint(-1, 1)
            up = np.random.randint(-1, 1)
            if agent2.infected_time >= self.incubation_time + low \
                    and agent2.infected_time <= self.contagion_time + up:
                contagion_test = np.random.random()
                # 
                if contagion_test <= self.contagion_rate:
                    agent1.status = Status.Infected
                    agent1.infected_status = InfectionSeverity.Asymptomatic

        self.callback('post_contact', agent1, agent2)

    def get_statistics(self, kind='all'):
        if self.statistics is None:
            self.statistics = {}
            for quintile in [0, 1, 2, 3, 4]:
                self.statistics['Q{}'.format(quintile + 1)] = np.sum(
                    h.wealth for h in self.houses if h.social_stratum == quintile
                ) / self.total_wealth
            self.statistics['Business'] = np.sum([b.wealth for b in self.business]) / self.total_wealth
            self.statistics['Government'] = self.government.wealth / self.total_wealth

            for status in Status:
                self.statistics[status.name] = np.sum(
                    [1 for a in self.population if a.status == status]) / self.population_size

            for infected_status in filter(lambda x: x != InfectionSeverity.Exposed, InfectionSeverity):
                self.statistics[infected_status.name] = np.sum([1 for a in self.population if
                                                                a.infected_status == infected_status and
                                                                a.status != Status.Death]) / self.population_size

        return self.filter_stats(kind)
    
    def check_mode(self):
        if self.mode == 'normal':
            # no masks
            self.contagion_rate = self.original_contagion_rate
            self.contagion_distance = self.original_contagion_distance
            # no tests
            self.tests = False
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False 
            
        if self.mode == 'masks':
            # change contagion rate
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            #no tests
            self.tests = False
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False 
                
        if self.mode == 'tests':
            # turn on testing
            self.tests = True
            # no masks
            self.contagion_rate = self.original_contagion_rate
            self.contagion_distance = self.original_contagion_distance
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False 
            
        if self.mode == 'maskstests':
            # change contagion rate
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # turn on testing
            self.tests = True
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False 
            
        if self.mode == 'lockdown':
            # original contagion rate
            self.contagion_rate = self.original_contagion_rate
            self.contagion_distance = self.original_contagion_distance
            # no testing
            self.tests = False
            # no lockdown
            self.lockdown = True
            # no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False
            
        if self.mode == 'contact_tracing':
            # no masks
            self.contagion_rate = self.original_contagion_rate
            self.contagion_distance = self.original_contagion_distance
            # no tests
            self.tests = False
            # no lockdown
            self.lockdown = False
            # contact tracing
            self.contact_tracing == True
            # mobility not limited
            self.lim_mobility = False 
        
        if self.mode == 'lim_mobility':
            # no masks
            self.contagion_rate = self.original_contagion_rate
            self.contagion_distance = self.original_contagion_distance
            # no tests
            self.tests = False
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == False
            # mobility limited
            self.lim_mobility = True 
            
        if self.mode == 'stage1':
            '''only masks '''
            # masks
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # no tests
            self.tests = False
            # no lockdown
            self.lockdown = False
            #no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False 
            
        if self.mode == 'stage2':
            '''masks and tests '''
            # masks
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # no tests
            self.tests = True
            # no lockdown
            self.lockdown = False
            #no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False
            
        if self.mode == 'stage3':
            '''masks, tests and contact tracing'''
            # masks
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # tests
            self.tests = True
            # no lockdown
            self.lockdown = False
            # no contact tracing
            self.contact_tracing == True
            # mobility not limited
            self.lim_mobility = False 
            
        if self.mode == 'stage4':
            '''masks, tests and contact tracing and limited mobility'''
            # masks
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # tests
            self.tests = True
            # no lockdown
            self.lockdown = False
            # contact tracing
            self.contact_tracing == True
            # mobility limited
            self.lim_mobility = True 
        
        if self.mode == 'stage5':
            '''lockdown'''
            # no masks
            self.contagion_rate = self.mask_rate
            self.contagion_distance = self.mask_distance
            # no tests
            self.tests = False
            # lockdown
            self.lockdown = True
            #no contact tracing
            self.contact_tracing == False
            # mobility not limited
            self.lim_mobility = False
            
        return
    
    def change_mode(self, action):
        if action == 0:
            self.mode = 'stage1'
        elif action == 1:
            self.mode = 'stage2'
        elif action == 2:
            self.mode = 'stage3'
        elif action == 3:
            self.mode = 'stage4'
        elif action == 4:
            self.mode = 'stage5'
        print(action)
    
    def check_policy(self):
        if self.policy == 'normal':
            return
        
        if self.policy == 'simplepolicy1':
            day = simulation_day(self.iteration)
            if day >= 28:
                self.mode = 'lockdown'
            else:
                self.mode = 'masks'
            return
    
        if self.policy == 'increasing':
            day = simulation_day(self.iteration)
            if day < 9:
                self.mode = 'normal'
            elif day < 19:
                self.mode = 'stage1'
            elif day < 29:
                self.mode = 'stage2'
            elif day < 39:
                self.mode = 'stage3'
            elif day < 49:
                self.mode = 'stage4'
            elif day >= 49:
                self.mode = 'stage5'
            return
        
        if self.policy == 'decreasing':
            day = simulation_day(self.iteration)
            if day < 9:
                self.mode = 'stage5'
            elif day < 19:
                self.mode = 'stage4'
            elif day < 29:
                self.mode = 'stage3'
            elif day < 39:
                self.mode = 'stage2'
            elif day < 49:
                self.mode = 'stage1'
            elif day >= 49:
                self.mode = 'normal'
            return
        
        if self.policy == 'hospcap':
            hosp_cap = np.sum([1 for a in self.population if a.infected_status == InfectionSeverity.Severe ])
            if hosp_cap == 0:
                self.mode = 'stage1'
            if hosp_cap == 1:
                self.mode = 'stage3'
            if hosp_cap == 2:
                self.mode = 'stage5'
            return
        
        return
        
    def test_contacts(self):
        for pair in self.traced_contacts:
            ai = self.population[pair[0]]
            aj = self.population[pair[1]]
            if ai.infected_status == InfectionSeverity.Symptomatic and ai.infected_time == 6:
                if aj.status == Status.Infected:
                    aj.isolated = True
                    aj.protection_known = True
            elif aj.infected_status == InfectionSeverity.Symptomatic and aj.infected_time == 6:
                if ai.status == Status.Infected:
                    ai.isolated = True
                    ai.protection_known = True
                
  
    def executeRL(self, action):
        # set new mode 
        if action == 0:
            self.mode = 'stage1'
        elif action == 1:
            self.mode = 'stage2'
        elif action == 2:
            self.mode = 'stage3'
        elif action == 3:
            self.mode = 'stage4'
        elif action == 4:
            self.mode = 'stage5'
        self.check_mode()
        
        self.observation = np.zeros(5)
        self.reward = 0
        
        # execute 10 days = 240 iterations
        for i in range(240):  

            self.iteration += 1
            if self.callback('on_execute', self):
                return
            # sleep at night
            if not new_day(self.iteration) and bed_time(self.iteration):
                continue

            bed = bed_time(self.iteration)
            work = work_time(self.iteration)
            free = free_time(self.iteration)
            lunch = lunch_time(self.iteration)
            new_dy = new_day(self.iteration)
            work_dy = work_day(self.iteration)
            new_mth = new_month(self.iteration)
            day = simulation_day(self.iteration)

            for agent in filter(lambda x: x.status != Status.Death, self.population):
                # quarantine all isolated people
                if agent.isolated == True:
                    agent.move_to_home()
                    moving = True
                else:
                    moving = False
            
                #if not self.callback('on_person_move', agent):
                if self.lockdown == True:
                    if agent.house is not None:
                        agent.house.checkin(agent)
                
                if not self.lockdown:
                    if not moving:
                        if bed:
                            agent.move_to_home()

                        elif lunch or free or not work_dy:
                            if not self.lim_mobility == True:
                                agent.move_freely()

                        elif work_dy and work:
                            agent.move_to_work()
    
                    
                    

                self.callback('post_person_move', agent)



                if new_dy:
                    agent.update()
                
                    # trace contacts if desired
                    if self.contact_tracing == True:
                        self.test_contacts()
                        self.traced_contacts = []
                    


                if agent.infected_status == InfectionSeverity.Asymptomatic:
                    for bus in filter(lambda x: x != agent.employer, self.business):
                        if distance(agent, bus) <= self.business_distance:
                            bus.supply(agent)
            for bus in filter(lambda b: b.open, self.business):
                if new_dy:
                    bus.update()

                if self.iteration > 1 and new_mth:
                    bus.accounting()
            for house in filter(lambda h: h.size > 0, self.houses):
                if new_dy:
                    house.update()

                if self.iteration > 1 and new_mth:
                    house.accounting()
            if new_dy:
                self.government.update()
                self.healthcare.update()

            if self.iteration > 1 and new_mth:
                self.government.accounting()

            contacts = []
            for i in np.arange(0, self.population_size):
                for j in np.arange(i + 1, self.population_size):
                    ai = self.population[i]
                    aj = self.population[j]
                    if ai.status == Status.Death or ai.status == Status.Death:
                        continue

                    if distance(ai, aj) <= self.contagion_distance:
                        contacts.append((i, j))
                        self.traced_contacts.append((i,j))

            for pair in contacts:
                ai = self.population[pair[0]]
                aj = self.population[pair[1]]
                self.contact(ai, aj)
                self.contact(aj, ai)

            self.statistics = None

            self.callback('post_execute', self)
        
        # compute reward of 10th day
        sev = np.sum([1 for a in self.population if
                      a.infected_status == InfectionSeverity.Severe])/self.population_size
        a = 0.4
        b= -0.1
        p = 1.5
        self.reward += a*((self.critical_limit-sev)/self.critical_limit)+b*(action/4)**p     
        
        # observation at day 10
        self.statistics = None
        self.get_statistics(kind='info')
        incidence = 0         
        incidence = np.sum([1 for a in self.population if
                            a.isolated == True or
                            a.infected_status == InfectionSeverity.Severe])
        quarantined = np.sum([1 for a in self.population if
                                a.isolated == True])/self.population_size*100
        incidence = incidence/self.population_size *100
        self.observation[0] = incidence
        self.observation[1] = self.statistics['Severe']*100
        self.observation[2] = quarantined
        self.observation[3] = self.statistics['Death']*100
        known_immunity = np.sum([1 for a in self.population if 
                               a.status == Status.Recovered_Immune and
                               a.protection_known == True])/self.population_size*100
        self.observation[4] = known_immunity
            
        self.observation = self.observation.astype(float)
        
        return self.observation, self.reward



 