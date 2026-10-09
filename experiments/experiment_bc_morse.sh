#! /bin/bash
# MORSE with BC as the base learner. Extra arguments (--flag=value) override the defaults below, e.g.
#   bash experiment_bc_morse.sh --algorithm.morse_conflict_mode=none --runner.exp_name="MoE"
# RL-X rejects a flag that is given twice, so a default is dropped when the same flag is passed.
# See experiment_morse_ablations.sh for the ablations reported in the paper.

DATASET=${DATASET:-data/offline_data/all_robots_suboptimal_70_forward_1m}
COMMANDS=${COMMANDS:-command_forward.txt}

DEFAULTS=(
    --algorithm.name="uni_bc_morse.bc"
    --algorithm.nr_offline_updates=100000
    --algorithm.batch_size=1024
    --algorithm.actor_lr=3e-4
    --algorithm.max_grad_norm=5.0
    --algorithm.evaluation_episodes=1
    --algorithm.evaluation_frequency=100
    --algorithm.save_latest_frequency=10000
    --algorithm.determine_fastest_cpu_for_gpu=False
    --algorithm.device="gpu"
    --algorithm.dataset_path="$DATASET"
    --algorithm.eval_stochastic=False
    --environment.name="multi_robot_offline"
    --environment.nr_envs=16
    --environment.async_skip_percentage=0.0
    --environment.cycle_cpu_affinity=True
    --environment.seed=0
    --environment.nr_eval_envs=16
    --environment.test_command_path="$COMMANDS"
    --runner.mode="train"
    --runner.track_console=False
    --runner.track_tb=True
    --runner.track_wandb=True
    --runner.save_model=True
    --runner.wandb_entity="your-wandb-entity"
    --runner.project_name="cross-embodiment-offline"
    --runner.exp_name="MORSE-BC"
)

ARGS=()
for default in "${DEFAULTS[@]}"; do
    overridden=0
    for arg in "$@"; do
        [ "${arg%%=*}" = "${default%%=*}" ] && overridden=1
    done
    [ $overridden = 0 ] && ARGS+=("$default")
done

python experiment.py "${ARGS[@]}" "$@"
