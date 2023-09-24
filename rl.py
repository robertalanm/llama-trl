import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import pandas as pd
from reward.open_assistant import OpenAssistantRewardModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

checkpoint = "robertmyers/targon-7b"
revision = "v1.1.8"



# dataset
df = pd.read_parquet("./data/aa.parquet")

observation_list = [{"input": prompt} for prompt in df["prompt"]][:100]

print( df["prompt"])

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision, torch_dtype="auto")

model.to("cuda")

class MyRLEnv(TextRLEnv):
    def __init__(self, model, tokenizer, observation_input, max_length, compare_sample):
        super().__init__(model, tokenizer, observation_input, max_length, compare_sample)
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = OpenAssistantRewardModel("cuda")
        
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        total_reward = []
        output = ""
        predicted_list = [p[0] for p in predicted_list if p != ""]

        rewards = self.reward_model.get_rewards(input_item['input'], predicted_list, "test")

        
        if finish:
            reward = [1] * len(predicted_list)  # calculate reward score base on predicted_list
        return rewards[0].item()


# observaton_list = [{"input":"explain how attention work in seq2seq model"}]
# env = MyRLEnv(model, tokenizer, observation_input=observation_list, max_length=2048, compare_sample=2)
env = MyRLEnv(model, tokenizer, observation_list, 2048, 2)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)
# print(observation_list[0]['input'])
# print(actor.predict(observation_list[0]))

n_episodes = 1000
max_episode_len = 200  # max sentence length

for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0.01
    t = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, pred = env.step(action)
        print(action)
        print(reward)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())



print(actor.predict(observation_list[0]))