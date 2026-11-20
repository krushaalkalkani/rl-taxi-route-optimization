"""taxi_rl — reinforcement learning for taxi dispatch & routing on real road networks."""

from gymnasium.envs.registration import register

register(
    id="CityTaxi-v0",
    entry_point="taxi_rl.envs.city_taxi:CityTaxiEnv",
)

__version__ = "0.1.0"
