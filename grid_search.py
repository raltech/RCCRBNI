import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func, scatter_reward_function, diversity_reward_function, more_cell_reward_func
from utils.score_func import span_score_func, scatter_matrix_score_func, qr_rank_score_func, RREF_score_func, relevance_score_func
from utils.helper import load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.thompson_sampling import TSAgent
from agent.sarsa_agent import SARSAAgent
from agent.ucb1_agent import UCB1Agent
from agent.sarsa_ucb1_agent import SARSAUCB1Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import itertools

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def grid_search(agent_class, hyperparameter_ranges, env, data, n_search, log_freq, n_avg_itr, score_func):
    best_score = float('-inf')
    best_hyperparameters = None
    best_score_hist = None

    for hyperparameters in itertools.product(*hyperparameter_ranges.values()):
        hyperparameters_dict = dict(zip(hyperparameter_ranges.keys(), hyperparameters))

        print(f"hyperparameters: {hyperparameters_dict}")
        
        score_hist_list = []

        for avg_itr in range(n_avg_itr):
            print(f"avg_itr: {avg_itr}/{n_avg_itr}")
            env.reset()
            agent = agent_class(env, **hyperparameters_dict)

            score_hist = []

            for i in tqdm(range(n_search)):
                state = env.state
                if i % log_freq == 0:
                    good_actions = np.nonzero(agent.Q[state])[0]
                    approx_dict = env.get_est_dictionary()[good_actions,:]
                    if len(approx_dict) == 0:
                        score_hist.append(0)
                    else:
                        score_hist.append(score_func(approx_dict, dict_hat_count=env.dict_hat_count, relevance=data[-1]))
                action = agent.choose_action()
                next_state, reward, done = env.step(action2elec_amp(action, N_AMPLITUDES))
                agent.update(state, action, reward, next_state)

            score_hist_list.append(score_hist)

        avg_score = np.mean(score_hist_list, axis=0)[-1]
        print(f"avg_score: {avg_score}")

        if avg_score > best_score:
            best_score = avg_score
            best_hyperparameters = hyperparameters_dict
            best_score_hist = np.mean(score_hist_list, axis=0)

    return best_score, best_hyperparameters, best_score_hist

def main(n_search, log_freq, plot_histogram=False):
    score_func = relevance_score_func
    reward_func = diversity_reward_function
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    experiment = experiments[1]
    path = f"./data/{experiment}/dictionary"
    usage_path = f"/Volumes/Scratch/Users/ajphillips/gdm_streamed/{experiment}/estim" # TODO CHANGEME
    usage_path = f"./data/{experiment}/estim"
    # agent_list = ["RandomAgent", "EpsilonGreedyAgent", "DecayEpsilonGreedyAgent", 
    #               "TSAgent", "SARSAAgent", "UCB1Agent", "SARSAUCB1Agent"]
    agent_list = ["SARSAUCB1Agent"]

    # data: (dict, elecs, amps, elec_map, cell_ids, usage)
    data = load_dictionary(path, usage_path)
    # import pdb; pdb.set_trace()

    # calculate the span score of the dictionary
    baseline_score = score_func(data[0], dict_hat_count=np.ones(N_ELECTRODES*N_AMPLITUDES)*25, relevance=data[-1])
    print(f"score of the exhaustive dictionary: {baseline_score}")
    
    # average the scores for multiple runs
    avg_score_hist_list = []
    n_avg_itr = 10

    for agent_name in agent_list:
        print("\n========================================")
        print(f"Agent: {agent_name}")

        if agent_name == "RandomAgent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'epsilon': [1.0],
                'decay_rate': [1],
                'lr': [0.1]
            }
            agent_class = EpsilonGreedyAgent
        elif agent_name == "EpsilonGreedyAgent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'epsilon': [0.4, 0.6, 0.8, 0.9],
                'decay_rate': [1],
                'lr': [0.1]
            }
            agent_class = EpsilonGreedyAgent
        elif agent_name == "DecayEpsilonGreedyAgent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'epsilon': [1.0],
                'decay_rate': [1-10e-6, 1-10e-5, 1-10e-4],
                'lr': [0.1]
            }
            agent_class = EpsilonGreedyAgent
        elif agent_name == "TSAgent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'lr': [0.1, 0.01]
            }
            agent_class = TSAgent
        elif agent_name == "SARSAAgent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'epsilon': [0.1,0.2,0.3,0.4],
                'lr': [0.1]
            }
            agent_class = SARSAAgent
        elif agent_name == "UCB1Agent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'c': [0.1, 0.5, 1.0],
                'lr': [0.1]
            }
            agent_class = UCB1Agent
        elif agent_name == "SARSAUCB1Agent":
            hyperparameter_ranges = {
                'gamma': [0.9],
                'c': [0.5, 1.0, 2.0, 4.0],
                'lr': [0.1]
            }
            agent_class = SARSAUCB1Agent
        else:
            raise ValueError("Agent name not found")
        
        # initialize the environment
        env = SingleStateEnvironment(data, reward_func=reward_func, n_maxstep=N_EXAUSTIVE_SEARCH, 
                                    n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)

        best_score, best_hyperparameters, best_score_hist = grid_search(
            agent_class=agent_class,
            hyperparameter_ranges=hyperparameter_ranges,
            env=env,
            data=data,
            n_search=n_search,
            log_freq=log_freq,
            n_avg_itr=n_avg_itr,
            score_func=score_func
        )

        print(f"Best score for {agent_name}: {best_score}")
        print(f"Best hyperparameters for {agent_name}: {best_hyperparameters}")

        avg_score_hist_list.append(best_score_hist)

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
    fig.savefig(f"./assets/scores_{experiment}_{score_func.__name__}_{reward_func.__name__}_{n_search}_{n_avg_itr}.png")
    plt.show()

if __name__ == "__main__":
    main(n_search=250001, log_freq=50000)
