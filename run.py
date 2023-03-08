from env import SimulationEnv, FullStateSimulationEnv
from utils.reward_func import inverse_reward_func
from utils.score_func import span_score_func
from utils.helper import display_non_zero
from agent import EpsilonGreedyAgent, StatelessEpsilonGreedyAgent

def main():
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[0]}/dictionary"

    sim = FullStateSimulationEnv(path, reward_func=inverse_reward_func, score_func=span_score_func, n_maxstep=100000, n_elecs=512, n_amps=42)

    agent = StatelessEpsilonGreedyAgent(sim, epsilon=0.5, gamma=0.9, alpha=0.1)
    agent.run(n_episodes=50000)

    display_non_zero(agent.Q.reshape(-1,1))

if __name__ == "__main__":
    main()