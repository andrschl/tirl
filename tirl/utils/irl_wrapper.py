from gymnasium import Wrapper

class IRLRewardWrapper(Wrapper):
    def __init__(self, env, irl_model):
        super().__init__(env)
        self.irl_model = irl_model

    def step(self, action):
        obs_next, env_reward, done, info = self.env.step(action)
        current_obs, _ = self.env.get_observations()
        # IRL-based reward
        irl_reward = self.irl_model.get_reward(current_obs, action)
        # e.g. override the environment reward:
        # final_reward = mix_alpha*irl_reward + (1 - mix_alpha)*env_reward
        final_reward = irl_reward
        return obs_next, final_reward, done, info
