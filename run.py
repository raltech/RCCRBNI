import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func, scatter_reward_function, diversity_reward_function, more_cell_reward_func
from utils.score_func import span_score_func, scatter_matrix_score_func, qr_rank_score_func, RREF_score_func
from utils.helper import load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.thompson_sampling import TSAgent
from agent.sarsa_agent import SARSAAgent
from agent.ucb1_agent import UCB1Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def main(n_search, log_freq, plot_histogram=False):
    score_func = qr_rank_score_func
    reward_func = diversity_reward_function
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    experiment = experiments[1]
    path = f"./data/{experiment}/dictionary"
    usage_path = f"/Volumes/Scratch/Users/ajphillips/gdm_streamed/{experiment}/estim" # TODO CHANGEME
    agent_list = ["RandomAgent", "EpsilonGreedyAgent", "DecayEpsilonGreedyAgent", 
                  "TSAgent", "SARSAAgent", "UCB1Agent"]
    # agent_list = ["UCB1Agent"]

    # data: (dict, elecs, amps, elec_map, cell_ids, usage)
    data = load_dictionary(path, usage_path)

    # calculate the span score of the dictionary
    baseline_score = score_func(data[0], dict_hat_count=np.ones(data[0].shape[0])*25)
    print(f"Span score of the exhaustive dictionary: {baseline_score}")
    
    # average the scores for multiple runs
    avg_score_hist_list = []
    n_avg_itr = 1
    
    for agent_name in agent_list:
        print("\n========================================")
        print(f"Agent: {agent_name}")

        # keep track of the scores for multiple runs
        score_hist_list = []

        for avg_itr in range(n_avg_itr):
            print(f"Average iteration: {avg_itr+1}/{n_avg_itr}")

            # initialize the environment
            env = SingleStateEnvironment(data, reward_func=reward_func, n_maxstep=N_EXAUSTIVE_SEARCH, 
                                         n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)
            
            # initialize the agent
            if agent_name == "RandomAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=1.0, decay_rate=1, lr=0.1)
            elif agent_name == "EpsilonGreedyAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=0.8, decay_rate=1, lr=0.1)
            elif agent_name == "DecayEpsilonGreedyAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=1.0, decay_rate=1-10e-8, lr=0.1)
            elif agent_name == "TSAgent":
                agent = TSAgent(env, gamma=0.9, lr=0.1)
            elif agent_name == "SARSAAgent":
                agent = SARSAAgent(env, gamma=0.9, epsilon=0.8, lr=0.1)
            elif agent_name == "UCB1Agent":
                agent = UCB1Agent(env, gamma=0.9, c=2.0, lr=0.1)
            else:
                raise ValueError("Agent name not found")
            
            # keep track of the score
            score_hist = []

            # run the experiment
            for i in tqdm(range(n_search)):
                state = env.state
                if i % log_freq == 0:
                    good_actions = np.nonzero(agent.Q[state])[0]
                    # print(f"Number of good actions: {good_actions.shape}")
                    approx_dict = env.get_est_dictionary()[good_actions,:]
                    if len(approx_dict) == 0:
                        score_hist.append(0)
                    else:
                        score_hist.append(score_func(approx_dict, dict_hat_count=env.dict_hat_count))
                action = agent.choose_action()
                next_state, reward, done = env.step(action2elec_amp(action, N_AMPLITUDES))
                agent.update(state, action, reward, next_state)

            score_hist_list.append(score_hist)

        avg_score_hist_list.append(np.mean(score_hist_list, axis=0))

    # plot the span scores for each agent in the same plot
    fig, ax = plt.subplots()
    for i, agent_name in enumerate(agent_list):
        ax.plot(avg_score_hist_list[i], label=agent_name)
    ax.set_xlabel("Episode {0}k".format(log_freq//1000))
    ax.set_ylabel("Score")
    ax.set_title("Score")
    ax.legend()
    ax.axvline(x=N_EXAUSTIVE_SEARCH/log_freq, color="black", linestyle="--")
    ax.axhline(y=baseline_score, color="black", linestyle="--")
    fig.savefig(f"./assets/scores_{score_func.__name__}_{reward_func.__name__}_{n_search}_{n_avg_itr}.png")
    plt.show()

    if plot_histogram:
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
    main(n_search=500000, log_freq=10000)
