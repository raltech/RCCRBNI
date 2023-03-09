import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func
from utils.score_func import span_score_func, mag_cosine_sim_score_func, cosine_sim_score_func
from utils.helper import display_non_zero, get_dict_from_action_idx, load_dictionary, action2elec_amp
# from agent import EpsilonGreedyAgent, StatelessEpsilonGreedyAgent, StatelessExpDecayEpsilonGreedyAgent, StatelessTSAgend
from agent.epsilon_greedy import EpsilonGreedyAgent

def main():
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[0]}/dictionary"

    env = SingleStateEnvironment(path, reward_func=inverse_reward_func, score_func=span_score_func, n_maxstep=15000000, n_elecs=512, n_amps=42)
    agent = EpsilonGreedyAgent(env, gamma=0.8, epsilon=0.9, decay_rate=1-10e-7)
    agent.run(n_episodes=50000)

    display_non_zero(agent.Q, top_k=10)

    new_dict = []
    # choose state
    state = 0
    for a in np.nonzero(agent.Q[state])[0]:
        new_dict.append(get_dict_from_action_idx(a, env.n_amps, path))
    new_dict = np.array(new_dict)

    dict = load_dictionary(path)[0]
    print("Original dict: ", dict.shape)
    print("New dict: ", new_dict.shape)
    print("Original Span: ", span_score_func(dict))
    print("New Span: ", span_score_func(new_dict))
    print("Original Magnitude Cosine Similarity: ", cosine_sim_score_func(dict))
    print("New Magnitude Cosine Similarity: ", cosine_sim_score_func(new_dict))

if __name__ == "__main__":
    main()