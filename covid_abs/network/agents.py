from enum import Enum

import numpy as np

from covid_abs.agents import Agent, AgentType, InfectionSeverity, Status
from covid_abs.common import *


class EconomicalStatus(Enum):
    Active = 1
    Inactive = 0


class Business(Agent):
    """
    The container of Agent's attributes and status
    """
    def __init__(self, **kwargs):
        super(Business, self).__init__(**kwargs)
        self.employees = [] 
        self.num_employees = 0 
        self.type = AgentType.Business 
        self.incomes = 0.0 
        self.expenses = 0.0 
        #self.labor_expenses = {}
        self.stocks = 10
        self.sales = 0
        self.open = True 
        self.type = kwargs.get("type", AgentType.Business)
        self.price = kwargs.get("price", (self.social_stratum+1) * 12.0)
        self.fixed_expenses = kwargs.get('fixed_expenses', 0.0)

    def cash(self, value):
        self.wealth += value

    def hire(self, agent):
        if agent.status != Status.Death and agent.infected_status == InfectionSeverity.Asymptomatic:
            self.employees.append(agent)
            #self.labor_expenses[agent.id] = 0.0
            agent.employer = self
            self.num_employees += 1
            self.fixed_expenses += (agent.expenses / 720) * 24

    def fire(self, agent):
        if self.environment.callback('on_business_fire', self):
            return
        self.employees.remove(agent)
        #self.labor_expenses[agent.id] = None
        agent.employer = None
        agent.supply(agent.incomes)
        self.cash(-agent.incomes)
        self.num_employees -= 1
        self.fixed_expenses -= (agent.expenses / 720) * 24

    def demand(self, agent):
        """Expenses due to employee payments"""
        if self.environment.callback('on_business_demand', self):
            return
        labor = 0
        if agent in self.employees:
            #labor = self.labor_expenses[agent.id]
            if agent.status != Status.Death and agent.infected_status == InfectionSeverity.Asymptomatic:
                labor = agent.incomes
                agent.supply(labor)
            #self.labor_expenses[agent.id] = 0
        elif agent.type == AgentType.Healthcare:
            labor = agent.expenses
            agent.cash(labor)
        else:
            if agent.status != Status.Death and agent.infected_status == InfectionSeverity.Asymptomatic:
                labor = agent.expenses
                agent.supply(labor)

        self.cash(-labor)

        return labor

    def supply(self, agent):
        """Incomes due to selling product/service"""
        if self.environment.callback('on_business_supply', self):
            return
        qty = np.random.randint(1, 10)
        if qty > self.stocks:
            qty = self.stocks
        value = self.price * agent.social_stratum * qty
        if agent.type == AgentType.Person:
            agent.demand(value)
        else:
            agent.cash(-value)
        self.cash(value)
        self.incomes += value
        self.stocks -= qty
        self.sales += qty

    def checkin(self, agent):
        """Employee is working"""

        if self.environment.callback('on_business_checkin', self):
            return

        if self.type == AgentType.Business:
            self.stocks += 1
            self.cash(-agent.expenses/720)
            #self.labor_expenses[agent.id] += agent.income / 160
        elif self.type == AgentType.Healthcare:
            self.expenses += agent.expenses

    def taxes(self):
        """Expenses due to employee payments"""
        tax = self.environment.government.price * self.num_employees + self.incomes/20
        self.environment.government.cash(tax)
        self.cash(-tax)
        return tax

    def accounting(self):
        
        if self.environment.callback('on_business_accounting', self):
            return 
        
        if self.type == AgentType.Business:
            labor = 0.0
            for person in self.employees:
                labor += self.demand(person)
            tax = self.taxes()

            if 2 * (labor + tax) < self.incomes:
                unemployed = self.environment.get_unemployed()
                if len(unemployed) >0:
                    ix = np.random.randint(0, len(unemployed))
                    self.hire(unemployed[ix])
            elif (labor + tax) > self.incomes:
                ix = np.random.randint(0, self.num_employees)
                self.fire(self.employees[ix])
        elif self.type == AgentType.Healthcare:
            self.environment.government.demand(self)
        elif self.type == AgentType.Government:
            self.demand(self.environment.healthcare)
            for person in self.environment.get_homeless():
                self.demand(person)
            for person in self.environment.get_unemployed():
                self.demand(person)

        self.incomes = 0
        self.sales = 0

        self.environment.callback('post_business_accounting', self)

    def update(self):
        if self.environment.callback('on_business_update', self):
            return 
        
        if self.type != AgentType.Government:
            self.environment.government.cash(self.fixed_expenses/3)
            self.cash(-self.fixed_expenses) #The money get out of this local economy - this effect is represented by foreign markets (suppliers) and financial system
            
        else:
            # This represents the public spending in the local economy
            ix = np.random.randint(0, self.environment.total_business)
            self.environment.business[ix].supply(self)

        self.environment.callback('post_business_update', self)


class House(Agent):
    """
    The container of Agent's attributes and status
    """

    def __init__(self, **kwargs):
        super(House, self).__init__(**kwargs)
        self.homemates = []
        self.type = AgentType.House
        self.size = 0
        self.incomes = 0
        self.expenses = 0
        self.fixed_expenses = kwargs.get('fixed_expenses',0.0)
        self.type == AgentType.House

    def append_mate(self, agent):
        self.homemates.append(agent)
        self.wealth += agent.wealth
        self.size += 1
        agent.house = self
        x, y = np.random.normal(0.0, 0.25, 2)
        agent.x = int(self.x + x)
        agent.y = int(self.y + y)
        self.fixed_expenses += (agent.expenses / 720) * 24

    def remove_mate(self, agent):
        self.homemates.remove(agent)
        self.wealth -= agent.wealth/2
        self.size -= 1
        self.fixed_expenses -= (agent.expenses / 720) * 24

    def checkin(self, agent):
        if self.environment.callback('on_house_checkin', self):
            return
        self.demand(agent.expenses/720)

        self.environment.callback('post_house_checkin', self)

    def demand(self, value = 0.0):
        """Expense of consuming product/services"""
        if self.environment.callback('on_house_demand', self):
            return
        self.wealth -= value
        self.expenses += value
        self.environment.callback('post_house_demand', self)

    def supply(self, value = 0.0):
        """Income of work"""
        if self.environment.callback('on_house_supply', self):
            return

        self.wealth += value
        self.incomes += value

        self.environment.callback('post_house_supply', self)

    def accounting(self):
        if self.environment.callback('on_house_accounting', self):
            return 
        
        """Expenses due to employee payments"""
        tax = self.environment.government.price * len(self.homemates) + self.expenses/10
        self.environment.government.cash(tax)
        self.wealth -= tax
        self.incomes = 0
        self.expenses = 0

        self.environment.callback('post_house_accounting', self)

    def update(self):
        if self.environment.callback('on_house_update', self):
            return 
            
        #self.demand(self.fixed_expenses) # the money get out of the local economy
        self.environment.government.cash(self.fixed_expenses/10) # daily taxes


        self.environment.callback('post_house_update', self)


class Person(Agent):
    """
    The container of Agent's attributes and status
    """

    def __init__(self, **kwargs):
        super(Person, self).__init__(**kwargs)
        self.employer = kwargs.get("employer", None)
        self.house = kwargs.get("house", None)
        self.type = AgentType.Person
        self.economical_status = EconomicalStatus.Inactive
        self.incomes = kwargs.get("income", 0.0)
        self.expenses = kwargs.get("expense", 0.0)
        self.isolated = kwargs.get("isolated", False)

        if self.age > 16 and self.age <= 65:
            self.economical_status = EconomicalStatus.Active

    def is_unemployed(self):
        return self.employer is None and self.economical_status == EconomicalStatus.Active

    def is_homeless(self):
        return self.house is None

    def demand(self, value = 0.0):
        """Expense for product/services"""
        if self.house is not None:
            self.house.demand(value)
        else:
            self.wealth -= value

    def supply(self, value = 0.0):
        """Income for work"""
        if self.house is not None:
            self.house.supply(value)
        else:
            self.wealth += value

    def move_to_work(self):
        if self.environment.callback('on_person_move', self) or \
                self.environment.callback('on_person_move_to_work', self):
            return
        
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        if self.economical_status == EconomicalStatus.Active:
            if self.employer is not None and self.employer.open:
                x, y = np.random.normal(0.0, 0.25, 2)
                self.x = int(self.employer.x + x)
                self.y = int(self.employer.y + y)
                self.employer.checkin(self)
            elif self.employer is None:
                self.move_freely()

        self.environment.callback('post_person_move', self)
        self.environment.callback('post_person_move_to_work', self)

    def move_to_home(self):
        if self.environment.callback('on_person_move_to_home', self):
            return

        #if self.infected_status != InfectionSeverity.Asymptomatic:
        #    return

        if self.house is not None:
            self.house.checkin(self)
            x, y = np.random.normal(0.0, 0.25, 2)
            self.x = int(self.house.x + x)
            self.y = int(self.house.y + y)
        else:
            self.wealth -= self.incomes / 720
            self.move_freely()

        self.environment.callback('post_person_move_to_home', self)

    def move_freely(self):
        #if self.infected_status == InfectionSeverity.Symptomatic:
        #    return
        if self.environment.callback('on_person_move_freely', self):
            return

        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        x,y = np.random.normal(0, self.environment.amplitudes[self.status], 2)
        self.x = int(self.x + x)
        self.y = int(self.y + y)

        self.environment.callback('post_person_move_freely', self)

    def move_to(self, agent):
        if self.environment.callback('on_person_move_to', self, agent):
            return

        x, y = np.random.normal(0.0, 0.25, 2)
        self.x = int(agent.x + x)
        self.y = int(agent.y + y)

        agent.checkin(self)

        self.environment.callback('post_person_move_to', self)

    def check_balance(self, value):
        if self.house is not None:
            return value <= self.house.wealth
        else:
            return value <= self.wealth

    def update(self):
        """
        Update the status of the agent

        :param agent: an instance of agents.Agent
        """

        if self.environment.callback('on_person_update', self):
            return

        if self.status == Status.Death:
            return

        if self.status == Status.Infected:
            self.infected_time += 1

            ix = self.age // 10 - 1 if self.age > 10 else 0
            
            
           
            if self.infected_time == 6:
                if self.infected_status == InfectionSeverity.Asymptomatic:
                    symptomatic_test = np.random.random()
                    if symptomatic_test <= self.environment.symptomatic_rate:
                        self.infected_status = InfectionSeverity.Symptomatic
                        self.isolated = True
                        self.protection_known = True

            test_sub = np.random.random()

            if self.infected_status == InfectionSeverity.Symptomatic:
                if age_severe_probs[ix] > test_sub:
                    self.infected_status = InfectionSeverity.Severe
                    self.isolated = False
                    self.move_to(self.environment.healthcare)
            if self.infected_status == InfectionSeverity.Severe:
                stats = self.environment.get_statistics(kind='info')
                if stats['Severe']  >= self.environment.critical_limit:
                    self.status = Status.Death
                    self.infected_status = InfectionSeverity.Asymptomatic
                    if self.house is not None:
                        self.house.remove_mate(self)
                    else:
                        self.environment.government.cash(-self.expenses)

                        if self.employer is not None:
                            self.employer.fire(self)
                        else:
                            self.environment.government.cash(-self.expenses)
                self.environment.statistics = None
    
                death_test = np.random.random()
                if age_death_probs[ix] > death_test:
                    self.status = Status.Death
                    self.infected_status = InfectionSeverity.Asymptomatic
                    self.move_to_home()
                    return
            
                

            if self.infected_time > self.environment.recovering_time:
                self.infected_time = 0
                self.status = Status.Recovered_Immune
                self.infected_status = InfectionSeverity.Asymptomatic
                self.isolated = False
        
        # Random Tests
        if self.environment.tests == True:
            test_test = np.random.random()
            if test_test < self.environment.test_rate:
                covid_test = np.random.random()
                if covid_test <= self.environment.test_accuracy: 
                    if self.infected_status == InfectionSeverity.Asymptomatic and self.status == Status.Infected:
                        self.isolated = True
                        self.protection_known = True
                    if self.status == Status.Susceptible:
                        self.isolated = False
                else:
                    if self.status == Status.Susceptible:
                        self.isolated = True
                        self.protection_known
                        
        # Vaccination
        if self.environment.vaccination == True:
            vac_test =  np.random.random()
            if vac_test < self.environment.vac_rate:
                if self.status == Status.Susceptible or (self.status == Status.Infected and self.infected_status == InfectionSeverity.Asymptomatic):
                    self.status = Status.Recovered_Immune
                    self.protection_known = True

                            

        self.environment.callback('post_person_update', self)