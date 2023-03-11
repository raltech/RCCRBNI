import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func
from utils.score_func import span_score_func, mag_cosine_sim_score_func, cosine_sim_score_func
from utils.helper import display_non_zero, get_dict_from_action_idx, load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.thompson_sampling import TSAgent
from tqdm import tqdm
import matplotlib.pyplot as plt

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def main():
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[1]}/dictionary"

    env = SingleStateEnvironment(path, reward_func=inverse_reward_func, score_func=span_score_func, 
                                 n_maxstep=N_EXAUSTIVE_SEARCH, n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)
    # agent = EpsilonGreedyAgent(env, gamma=0.7, epsilon=0.8, decay_rate=1-10e-7, lr=0.1)
    agent = TSAgent(env, gamma=0.9, lr=0.1)
    log_freq = 10000
    new_dicts = agent.run(n_episodes=300000, log_freq=log_freq)

    # display_non_zero(agent.Q, top_k=10)

    dict = load_dictionary(path)[0]
    original_span = span_score_func(dict)
    original_cosine_sim = cosine_sim_score_func(dict)
    print("Original dict: ", dict.shape)
    print("Original Span: ", original_span)
    print("Original Cosine Similarity: ", original_cosine_sim)
    
    span_scores = []
    cosine_sim_scores = []
    for new_dict in new_dicts:
        span_scores.append(span_score_func(new_dict))
        cosine_sim_scores.append(cosine_sim_score_func(new_dict))
        print("New dict: ", new_dict.shape)
        print("New Span: ", span_scores[-1])
        print("New Cosine Similarity: ", cosine_sim_scores[-1])

    # plot the span score
    fig, ax = plt.subplots()
    ax.plot(span_scores, label="Span Score", color="blue")
    ax.plot(cosine_sim_scores, label="Cosine Similarity Score", color="orange")
    ax.set_xlabel("Episode {0}k".format(log_freq//1000))
    ax.set_ylabel("Score")
    if isinstance(agent, EpsilonGreedyAgent):
        ax.set_title("Epsilon Greedy Agent with gamma={}, epsilon={}, decay_rate={}, lr={}".format(agent.gamma, agent.epsilon, agent.decay_rate, agent.lr))
    elif isinstance(agent, TSAgent):
        ax.set_title("Thompson Sampling Agent with gamma={}, lr={}".format(agent.gamma, agent.lr))
    else:
        ax.set_title("Unknown Agent")
    ax.legend()
    ax.axvline(x=N_EXAUSTIVE_SEARCH/log_freq, color="red", linestyle="--")
    ax.axhline(y=original_span, color="blue", linestyle="--")
    ax.axhline(y=original_cosine_sim, color="orange", linestyle="--")
    plt.show()


if __name__ == "__main__":
    main()