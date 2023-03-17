import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func
from utils.score_func import span_score_func, mag_cosine_sim_score_func, cosine_sim_score_func, volume_score_func
from utils.helper import display_non_zero, get_dict_from_action_idx, load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.thompson_sampling import TSAgent
from tqdm import tqdm
import matplotlib.pyplot as plt

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def main(n_iterations, log_freq, plot_hist=False):
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[1]}/dictionary"
    agent_list = ["EpsilonGreedyAgent", "TSAgent"]

    # data: (dict, elecs, amps, elec_map, cell_ids)
    data = load_dictionary(path)

    # calculate the span score of the dictionary
    original_span_score = span_score_func(data[0])
    print(f"Span score of the dictionary: {original_span_score}")

    env = SingleStateEnvironment(data, reward_func=inverse_reward_func, score_func=span_score_func, 
                                 n_maxstep=N_EXAUSTIVE_SEARCH, n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)
    
    # keep track of the span score for each agent
    span_score_hist_list = []
    for agent_name in agent_list:
        if agent_name == "EpsilonGreedyAgent":
            agent = EpsilonGreedyAgent(env, gamma=0.7, epsilon=0.8, decay_rate=1-10e-7, lr=0.1)
        elif agent_name == "TSAgent":
            agent = TSAgent(env, gamma=0.9, lr=0.1)
        else:
            raise ValueError("Agent name not found")
        
        # keep track of the score
        span_score_hist = []

        # run the experiment
        for i in tqdm(range(n_iterations)):
            state = env.state
            if i % log_freq == 0 and i != 0:
                good_actions = np.nonzero(agent.Q[state])[0]
                approx_dict = env.get_est_dictionary()[good_actions,:]
                span_score_hist.append(span_score_func(approx_dict))
            action = agent.choose_action()
            next_state, reward, done = env.step(action2elec_amp(action, N_AMPLITUDES))
            agent.update(state, action, reward, next_state)

        span_score_hist_list.append(span_score_hist)

    # plot the span scores for each agent in the same plot
    fig, ax = plt.subplots()
    for i, agent_name in enumerate(agent_list):
        ax.plot(span_score_hist_list[i], label=agent_name)
    ax.set_xlabel("Episode {0}k".format(log_freq//1000))
    ax.set_ylabel("Score")
    ax.set_title("Span Score")
    ax.legend()
    ax.axvline(x=N_EXAUSTIVE_SEARCH/log_freq, color="black", linestyle="--")
    ax.axhline(y=original_span_score, color="black", linestyle="--")
    plt.show()
  
    # plot the span score
    # fig, ax = plt.subplots()
    # ax.plot(span_score_hist, label="Span Score", color="red")
    # ax.set_xlabel("Episode {0}k".format(log_freq//1000))
    # ax.set_ylabel("Score")
    # if isinstance(agent, EpsilonGreedyAgent):
    #     ax.set_title("Epsilon Greedy Agent with gamma={}, epsilon={}, decay_rate={}, lr={}".format(agent.gamma, agent.epsilon, agent.decay_rate, agent.lr))
    # elif isinstance(agent, TSAgent):
    #     ax.set_title("Thompson Sampling Agent with gamma={}, lr={}".format(agent.gamma, agent.lr))
    # else:
    #     ax.set_title("Unknown Agent")
    # ax.legend()
    # plt.show()

    if plot_hist:
        # plot the histogram
        occurrences = agent.get_n()[0]
        print("max: ", np.max(occurrences))

        enable_bin = False
        if enable_bin:
            # Define the number of bins you want
            n_bins = len(occurrences) // 5
            # Create the histogram with bin
            plt.hist(range(len(occurrences)), weights=occurrences, bins=n_bins)
        else:
            # Create the histogram
            plt.bar(range(len(occurrences)), occurrences)

        # Add labels and title
        plt.xlabel('Action index')
        plt.ylabel('Number of occurrences')
        plt.title('Histogram of action occurrences')
        plt.yscale('log')
        plt.show()

if __name__ == "__main__":
    main(n_iterations=500000, log_freq=10000)