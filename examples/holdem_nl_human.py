''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''
from rlcard.agents.random_agent import RandomAgent

import rlcard
from rlcard import models
from rlcard.agents.holdem_nl_human_agent import HumanAgent
from rlcard.utils.utils import print_card

# Make environment
# Set 'record_action' to True because we need it to print results
env = rlcard.make('no-limit-holdem', config={'record_action': True})

agent = models.load('nolimit_holdem_dqn').agents[0]
human_agent = HumanAgent(env.action_num)
# agent = RandomAgent(action_num=env.action_num)

env.set_agents([human_agent, agent])


while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1][-2]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     Table    ===============')
    print_card(env.get_perfect_information()['public_card'])

    print('===============     Cards all Players    ===============')
    print("your cards:")
    print_card(env.get_perfect_information()['hand_cards'][0])
    print("opponent cards:")
    for hands in env.get_perfect_information()['hand_cards'][1:]:
        print_card(hands)

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
