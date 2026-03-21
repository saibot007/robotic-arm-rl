from gymnasium.utils.env_checker import check_env
from src.arm_env import ArmEnv

env = ArmEnv()
check_env(env)
print("Environment passed check_env")
env.close()