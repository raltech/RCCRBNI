from envs import SimulationEnv
from reward_func import inverse_reward_func
from score_func import span_score_func
from agents import EpsilonGreedyAgent

def main():
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[0]}/dictionary"

    sim = SimulationEnv(path, reward_func=inverse_reward_func, score_func=span_score_func, n_maxstep=5000, n_elecs=512, n_amps=42)

    agent = EpsilonGreedyAgent(sim, epsilon=0.8, gamma=0.9, alpha=0.1)
    policy = agent.run(n_episodes=500)
    print(policy)

if __name__ == "__main__":
    main()