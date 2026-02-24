from rl_x.runner.runner import Runner


if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "cross_embodiment_offline_rl"])
    runner.run()