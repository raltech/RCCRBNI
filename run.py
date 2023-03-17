import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func, max_cosine_sim_reward_func
from utils.score_func import span_score_func, cosine_sim_score_func
from utils.helper import display_non_zero, get_dict_from_action_idx, load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.thompson_sampling import TSAgent
from agent.sarsa_agent import SARSAAgent
from tqdm import tqdm
import matplotlib.pyplot as plt

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25
PLOT_COLOR = ["r", "g", "b", "y", "m", "c", "k"]

def main(n_iterations, log_freq, plot_histogram):
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[0]}/dictionary"
    agent_list = ["RandomAgent", "EpsilonGreedyAgent", "DecayingEpsilonGreedyAgent",
                  "TSAgent", "SARSAAgent"]

    # data: (dict, elecs, amps, elec_map, cell_ids)
    data = load_dictionary(path)

    # calculate the span score of the dictionary
    original_span_score = span_score_func(data[0])
    print(f"Span score of the dictionary: {original_span_score}")

    env = SingleStateEnvironment(data, reward_func=inverse_reward_func, n_maxstep=N_EXAUSTIVE_SEARCH,
                                 n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)
    
    # keep track of the span score for each agent
    avg_span_score_hist = []
    for agent_name in agent_list:
        if agent_name == "RandomAgent":
            agent = EpsilonGreedyAgent(env, gamma=0.8, epsilon=1.0, decay_rate=1.0, lr=0.2)
        elif agent_name == "EpsilonGreedyAgent":
            agent = EpsilonGreedyAgent(env, gamma=0.8, epsilon=0.8, decay_rate=1.0, lr=0.2)
        elif agent_name == "DecayingEpsilonGreedyAgent":
            agent = EpsilonGreedyAgent(env, gamma=0.8, epsilon=1.0, decay_rate=1-10e-8, lr=0.2)
        elif agent_name == "TSAgent":
            agent = TSAgent(env, gamma=0.8, lr=0.2)
        elif agent_name == "SARSAAgent":
            agent = SARSAAgent(env, gamma=0.8, epsilon=0.8, decay_rate=1.0, lr=0.2)
        else:
            raise ValueError("Agent name not found")

        # average the span score over multiple runs
        span_score_hist_list = []
        for _ in tqdm(range(2)):
            # keep track of the score
            span_score_hist = []

            # reset the environment
            env.reset()

            # run the experiment
            for i in range(n_iterations):
                state = env.state
                if i % log_freq == 0 and i != 0:
                    action_values = agent.get_Q()
                    good_actions = np.nonzero(action_values)[0]
                    approx_dict = env.get_est_dictionary()[good_actions,:]
                    span_score_hist.append(span_score_func(approx_dict))
                action = agent.choose_action()
                next_state, reward, done = env.step(action2elec_amp(action, N_AMPLITUDES))
                agent.update(state, action, reward, next_state)

            span_score_hist_list.append(span_score_hist)

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

        # average the span score over multiple runs
        avg_span_score_hist.append(np.mean(span_score_hist_list, axis=0))

    # plot the span scores for each agent in the same plot
    fig, ax = plt.subplots()
    for i, agent_name in enumerate(agent_list):
        ax.plot(avg_span_score_hist[i], label=agent_name, color=PLOT_COLOR[i])
        # draw the vertical line at place where the score is the same as the original dictionary
        tmp = np.where(avg_span_score_hist[i]==original_span_score)[0]
        if len(tmp) > 0:
            ax.axvline(x=tmp[0], color=PLOT_COLOR[i] ,linestyle="--")
    ax.set_xlabel("Episode {0}k".format(log_freq//1000))
    ax.set_ylabel("Score")
    ax.set_title("Span Score")
    ax.legend()
    ax.axvline(x=N_EXAUSTIVE_SEARCH/log_freq, color="black", linestyle="--")
    ax.axhline(y=original_span_score, color="black", linestyle="--")
    plt.show()

if __name__ == "__main__":
    main(n_iterations=100001, log_freq=5000, plot_histogram=False)