from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph


class AirCargoProblem(Problem):

    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        # Create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            loads = []
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        # In order to load the cargo, we need the cargo and the plane
                        # to be at the the airport.
                        precond_pos = [expr("At({}, {})".format(cargo, airport)), expr(
                            "At({}, {})".format(plane, airport))]
                        precond_neg = []
                        # After a load, the cargo is in the plane.
                        effect_add = [expr("In({}, {})".format(cargo, plane))]
                        # After a load, the cargo is no longer "at" the airport.
                        effect_rem = [expr("At({}, {})".format(cargo, airport))]
                        loads.append(Action(expr("Load({}, {}, {})".format(cargo, plane, airport)),
                                            [precond_pos, precond_neg], [effect_add, effect_rem]))
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        # In order to unload the cargo, we need the cargo to be
                        # in the plane and the plane at the airport.
                        precond_pos = [expr("In({}, {})".format(cargo, plane)), expr(
                            "At({}, {})".format(plane, airport))]
                        precond_neg = []
                        # After an unload, the cargo is at the airport.
                        effect_add = [expr("At({}, {})".format(cargo, airport))]
                        # After an unload, the cargo is no longer in the plane.
                        effect_rem = [expr("In({}, {})".format(cargo, plane))]
                        unloads.append(Action(expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                                              [precond_pos, precond_neg], [effect_add, effect_rem]))
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            # In order to fly, the plane needs to be at the "from" airport.
                            precond_pos = [expr("At({}, {})".format(p, fr)), ]
                            precond_neg = []
                            # After the fly, the plane is at the "to" airport.
                            effect_add = [expr("At({}, {})".format(p, to))]
                            # After the fly, the plane is no longer at the "from" airport.
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # Creating the knowledge base allows us to to use Action.check_procond
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        # All actions that have their preconditions met in the current state.
        possible_actions = [a for a in self.actions_list if a.check_precond(kb, a.args)]
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        kb = PropKB()
        curr_state = decode_state(state, self.state_map)
        kb.tell(curr_state.pos_sentence())
        # If the given action cannot be done in the current state,
        # then we return the current state.
        if not action.check_precond(kb, action.args):
            return state
        # Do the action with the current knowledge base.
        action.act(kb, action.args)
        # Construct the new_state straight from the knowledge base.
        new_state = ""
        for state in self.state_map:
            new_state += 'T' if kb.ask_if_true(state) else 'F'
        return new_state

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_pg_setlevel(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the number of actions that must be carried
        out to achieve all goals.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_setlevel = pg.h_setlevel()
        return pg_setlevel

    def h_pg_maxlevel(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the max of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_maxlevel = pg.h_maxlevel()
        return pg_maxlevel

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # Count the goals that are not true in the current state.
        # This is fine for us because any one action can only complete one goal.
        count = len([s for (i, s) in enumerate(self.state_map)
                     if s in self.goal and node.state[i] == "F"])
        return count


def air_cargo_p1() -> AirCargoProblem:
    # Two cargos.
    cargos = ['C1', 'C2']
    # Two planes.
    planes = ['P1', 'P2']
    # Two airports.
    airports = ['JFK', 'SFO']
    # Things that are true in the initial state.
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')]
    # Everything else is not true.
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)')]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # Three cargos.
    cargos = ['C1', 'C2', 'C3']
    # Three planes.
    planes = ['P1', 'P2', 'P3']
    # Three airports.
    airports = ['JFK', 'SFO', 'ATL']
    # Things that are true in the initial state.
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)')]
    neg = []
    # Everything else is false in the initial state.
    for a in airports:
        for c in (cargos + planes):
            e = expr("At({}, {})".format(c, a))
            if e not in pos:
                neg.append(e)
    # No cargo is in any plane.
    for p in planes:
        for c in cargos:
            neg.append(expr("In({}, {})".format(c, p)))
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # Four cargos.
    cargos = ['C1', 'C2', 'C3', 'C4']
    # Two planes
    planes = ['P1', 'P2']
    # Four airports
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    # Things that are true in the initial state.
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')]
    neg = []
    # Everything else is false in the initial state.
    for a in airports:
        for c in (cargos + planes):
            e = expr("At({}, {})".format(c, a))
            if e not in pos:
                neg.append(e)
    # No cargo starts in any plane.
    for p in planes:
        for c in cargos:
            neg.append(expr("In({}, {})".format(c, p)))
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)
