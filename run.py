import numpy as np
from env.single_state_env import SingleStateEnvironment
from utils.reward_func import inverse_reward_func, diversity_reward_function, more_cell_reward_func
from utils.score_func import span_score_func, scatter_matrix_score_func, qr_rank_score_func, RREF_score_func, relevance_score_func
from utils.helper import load_dictionary, action2elec_amp
from agent.epsilon_greedy import EpsilonGreedyAgent
from agent.sarsa_agent import SARSAAgent
from agent.ucb1_agent import UCB1Agent
from agent.sarsa_ucb1_agent import SARSAUCB1Agent
from tqdm import tqdm
import matplotlib.pyplot as plt

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def main(n_search, log_freq, experiment, score_func, reward_func, n_avg_itr):
    score_func = score_func
    reward_func = reward_func
    experiment = experiment
    n_avg_itr = n_avg_itr
    path = f"./data/{experiment}/dictionary"
    usage_path = f"/Volumes/Scratch/Users/ajphillips/gdm_streamed/{experiment}/estim" # TODO CHANGEME
    usage_path = f"./data/{experiment}/estim"
    agent_list = ["RandomAgent", "EpsilonGreedyAgent", "DecayEpsilonGreedyAgent", 
                  "UCB1Agent", "SARSAAgent", "SARSAUCB1Agent"]

    # data: (dict, elecs, amps, elec_map, cell_ids, usage)
    data = load_dictionary(path, usage_path)

    # calculate the span score of the dictionary
    baseline_score = score_func(data[0], dict_hat_count=np.ones(N_ELECTRODES*N_AMPLITUDES)*25, relevance=data[-1])
    print(f"score of the exhaustive dictionary: {baseline_score}")
    
    # average the scores for multiple runs
    avg_score_hist_list = []
    
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
            if agent_name == "RandomAgent" and avg_itr == 1:
                break
            # initialize the agent
            if agent_name == "RandomAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=1.0, decay_rate=1, lr=0.1)
            elif agent_name == "EpsilonGreedyAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=0.6, decay_rate=1, lr=0.1)
            elif agent_name == "DecayEpsilonGreedyAgent":
                agent = EpsilonGreedyAgent(env, gamma=0.9, epsilon=1.0, decay_rate=1-10e-6, lr=0.1)
            elif agent_name == "SARSAAgent":
                agent = SARSAAgent(env, gamma=0.9, epsilon=0.4, lr=0.1)
            elif agent_name == "UCB1Agent":
                agent = UCB1Agent(env, gamma=0.9, c=1.0, lr=0.1)
            elif agent_name == "SARSAUCB1Agent":
                agent = SARSAUCB1Agent(env, gamma=0.9, c=1.0, lr=0.1)
            else:
                raise ValueError("Agent name not found")
            
            # keep track of the score
            score_hist = []

            # run the experiment
            for i in tqdm(range(n_search)):
                state = env.state
                if i % log_freq == 0:
                    good_actions = np.nonzero(agent.Q[state])[0]
                    approx_dict = env.get_est_dictionary()[good_actions,:]
                    if len(approx_dict) == 0:
                        score_hist.append(0)
                    else:
                        score_hist.append(score_func(approx_dict, dict_hat_count=env.dict_hat_count, relevance=data[-1]))
                    if score_hist[-1] > baseline_score*3:
                        print("\nFound a 3x better dictionary at step {}. Score: {}".format(i, score_hist[-1]))
                        # append the score_hist with the same score for the rest of the search
                        score_hist.extend([score_hist[-1]]*((n_search-i)//log_freq))
                        break
                action = agent.choose_action()
                next_state, reward, done = env.step(action2elec_amp(action, N_AMPLITUDES))
                agent.update(state, action, reward, next_state)

            score_hist_list.append(score_hist)
        avg_score_hist_list.append(np.mean(score_hist_list, axis=0))

    # plot the span scores for each agent in the same plot
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 20})
    for i, agent_name in enumerate(agent_list):
        if agent_name == "RandomAgent":
            ax.plot(avg_score_hist_list[i], label="QL,Random", linewidth=3)
        elif agent_name == "EpsilonGreedyAgent":
            ax.plot(avg_score_hist_list[i], label="QL,$\epsilon$-Greedy", linewidth=3)
        elif agent_name == "DecayEpsilonGreedyAgent":
            ax.plot(avg_score_hist_list[i], label="QL,Decaying-$\epsilon$-Greedy", linewidth=3)
        elif agent_name == "UCB1Agent":
            ax.plot(avg_score_hist_list[i], label="QL,UCB1", linewidth=3)
        elif agent_name == "SARSAAgent":
            ax.plot(avg_score_hist_list[i], label="SARSA,$\epsilon$-Greedy", linewidth=3)
        elif agent_name == "SARSAUCB1Agent":
            ax.plot(avg_score_hist_list[i], label="SARSA,UCB1", linewidth=3)
        
    ax.figure.set_size_inches(20, 15)
    ax.set_xlabel("Step {0}k".format(log_freq//1000), fontsize=28)
    ax.set_ylabel("Score", fontsize=28)
    ax.set_title(f"Scores of each agent on {experiment} with {reward_func.__name__}", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    ax.set_ylim(0, baseline_score*2.7)
    ax.axvline(x=N_EXAUSTIVE_SEARCH/log_freq, color="black", linestyle="--")
    ax.axhline(y=baseline_score, color="black", linestyle="--")
    fig.savefig(f"./assets/scores/scores_{experiment}_{score_func.__name__}_{reward_func.__name__}_{n_search}_{n_avg_itr}.png", dpi=300)

if __name__ == "__main__":
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    n_search = 520001
    log_freq = 20000
    n_avg_itr = 2
    print("Experiment 1: {}_{}_{}".format(experiments[0], relevance_score_func.__name__, diversity_reward_function.__name__))
    main(n_search=n_search, log_freq=log_freq, experiment=experiments[0], score_func=relevance_score_func, reward_func=diversity_reward_function, n_avg_itr=n_avg_itr)

    # print("Experiment 2: {}_{}_{}".format(experiments[1], relevance_score_func.__name__, diversity_reward_function.__name__))
    # main(n_search=n_search, log_freq=log_freq, experiment=experiments[1], score_func=relevance_score_func, reward_func=diversity_reward_function, n_avg_itr=n_avg_itr)
    
    # print("Experiment 3: {}_{}_{}".format(experiments[0], relevance_score_func.__name__, inverse_reward_func.__name__))
    # main(n_search=n_search, log_freq=log_freq, experiment=experiments[0], score_func=relevance_score_func, reward_func=inverse_reward_func, n_avg_itr=n_avg_itr)
    
    # print("Experiment 4: {}_{}_{}".format(experiments[1], relevance_score_func.__name__, inverse_reward_func.__name__))
    # main(n_search=n_search, log_freq=log_freq, experiment=experiments[1], score_func=relevance_score_func, reward_func=inverse_reward_func, n_avg_itr=n_avg_itr)
    
    # print("Experiment 5: {}_{}_{}".format(experiments[0], relevance_score_func.__name__, more_cell_reward_func.__name__))
    # main(n_search=n_search, log_freq=log_freq, experiment=experiments[0], score_func=relevance_score_func, reward_func=more_cell_reward_func, n_avg_itr=n_avg_itr)
    
    # print("Experiment 6: {}_{}_{}".format(experiments[1], relevance_score_func.__name__, more_cell_reward_func.__name__))
    # main(n_search=n_search, log_freq=log_freq, experiment=experiments[1], score_func=relevance_score_func, reward_func=more_cell_reward_func, n_avg_itr=n_avg_itr)
