from enum import Enum
import time
import numpy as np
from copy import deepcopy
from rlcard.games.limitholdem.game import LimitholdemGame
from rlcard.games.nolimitholdem import montecarlo_python

from rlcard.games.nolimitholdem.dealer import NolimitholdemDealer as Dealer
from rlcard.games.nolimitholdem.player import NolimitholdemPlayer as Player
from rlcard.games.nolimitholdem.judger import NolimitholdemJudger as Judger
from rlcard.games.nolimitholdem.round import NolimitholdemRound as Round, Action


class Stage(Enum):

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class NolimitholdemGame(LimitholdemGame):

    def __init__(self, allow_step_back=False, num_players=2):
        ''' Initialize the class nolimitholdem Game
        '''
        self.allow_step_back = allow_step_back

        # small blind and big blind
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # config players
        self.num_players = num_players
        self.init_chips = 100

    def init_game(self):
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = Dealer()

        # Initilize two players to play the game
        self.players = [Player(i, self.init_chips) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger()

        # Deal cards to each  player to prepare for the first round
        for i in range(2 * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        # Initilize public cards
        self.public_cards = []
        self.pot = 0
        self.stage = Stage.PREFLOP

        # Randomly choose a big blind and a small blind
        s = np.random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].bet(chips=self.big_blind)
        self.players[s].bet(chips=self.small_blind)

        # The player next to the small blind plays the first
        self.game_pointer = (b + 1) % self.num_players

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(self.num_players, self.big_blind)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_nolimit_legal_actions(players=self.players)

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if action not in self.get_legal_actions():
            raise Exception('Action not allowed')

        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            self.history.append((r, b, r_c, d, p, ps))

        # Then we proceed to the next round
        self.pot = np.sum([player.in_chips for player in self.players])
        self.game_pointer = self.round.proceed_round(self.players, action, self.pot)

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 3 cards
            if self.round_counter == 0:
                self.stage = Stage.FLOP
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
            # For the following rounds, we deal only 1 card
            elif self.round_counter == 1:
                self.stage = Stage.TURN
                self.public_cards.append(self.dealer.deal_card())
            elif self.round_counter == 2:
                self.stage = Stage.RIVER
                self.public_cards.append(self.dealer.deal_card())

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_state(self, player):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player].get_state(self.public_cards, chips, legal_actions)
        state['stakes'] = [self.players[i].remained_chips for i in range(self.num_players)]
        state['current_player'] = self.game_pointer
        state['pot'] = self.pot
        state['stage'] = self.stage

        simulator = montecarlo_python.MonteCarlo()
        simulator.run_montecarlo(original_player_card_list=[[card[::-1] for card in state["hand"]]], original_table_card_list=[card[::-1] for card in state["public_cards"]], player_amount=2, maxRuns=1000, timeout=time.time() + 1, ghost_cards='', opponent_range=1, ui="")
        state['equity'] = simulator.equity

        return state

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players = self.history.pop()
            return True
        return False

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        hands = [p.hand + self.public_cards if p.status == 'alive' else None for p in self.players]
        chips_payoffs = self.judger.judge_game(self.players, hands)
        return chips_payoffs

    @staticmethod
    def get_action_num():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 4 actions (call, raise, check and fold)
        '''
        return len(Action)


#if __name__ == "__main__":
#    game = NolimitholdemGame()
#
#    while True:
#        print('New Game')
#        state, game_pointer = game.init_game()
#        print(game_pointer, state)
#        i = 1
#        while not game.is_over():
#            i += 1
#            legal_actions = game.get_legal_actions()
#            # if i == 3:
#            #     print('Step back')
#            #     print(game.step_back())
#            #     game_pointer = game.get_player_id()
#            #     print(game_pointer)
#            #     legal_actions = game.get_legal_actions()
#
#            action = np.random.choice(legal_actions)
#            # action = input()
#            # if action != 'call' and action != 'fold' and action != 'check':
#            #     action = int(action)
#            print(game_pointer, action, legal_actions)
#            state, game_pointer = game.step(action)
#            print(game_pointer, state)
#
#        print(game.get_payoffs())
