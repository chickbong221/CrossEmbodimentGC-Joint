# Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets

This repository provides the implementation for the paper:

<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://haruki-abe.github.io/cross_embodiment_offline_rl_website/" target="_blank">
      <b> Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets </b>
      </a>
      <br>
      Haruki Abe, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada
      <br>
      <em>International Conference on Learning Representations(ICLR)</em>, 2026
      <br>
      <a href="https://arxiv.org/abs/2602.18025">arXiv</a> /
      <a href="https://haruki-abe.github.io/cross_embodiment_offline_rl_website/" target="_blank">Project page</a> /
      <a href="https://huggingface.co/datasets/haruki-abe/cross_embodiment_offline_rl" target="_blank">Dataset</a>
    <br>
</td>

## Installation
1. Install RL-X

Default installation for a Linux system with a NVIDIA GPU.
For other configurations, see the RL-X [documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).
```bash
conda create -n cross_offline_rl python=3.11.4 -y
conda activate cross_offline_rl
git clone https://github.com/haruki-abe/RL-X.git
cd RL-X
git checkout rlx-cross-v1.0.0
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

2. Install the project
```bash
git clone https://github.com/haruki-abe/cross_embodiment_offline_rl.git
cd cross_embodiment_offline_rl
pip install -e .
```

## Download cross-embodiment datasets

> [!WARNING]
> These datasets are large.  
> Downloading all datasets may require **500GB+** of free disk space.  
> Downloading a single dataset may require **80GB+** of free disk space.  
> Downloads can take a long time depending on your network and storage speed.

> [!NOTE]
> If `hf` is not available, install the Hugging Face CLI (`huggingface_hub`) first.

### 1) Download all datasets

```bash
cd data
hf download haruki-abe/cross_embodiment_offline_rl \
  --repo-type dataset \
  --include "offline_data/**" \
  --local-dir .
```

### 2) Download selected datasets only
```bash
cd data
hf download haruki-abe/cross_embodiment_offline_rl \
  --repo-type dataset \
  --include "offline_data/all_robots_suboptimal_70_forward_1m/**" \
  --local-dir .
```


## Training a model
1. Set your W&B entity
```bash
# edit this file and set your own entity
vim experiments/wandb_entity.txt
bash experiments/apply_wandb_entity.sh
```

2. Run training
```bash
cd experiments
bash experiment_iql_eg.sh      # train IQL + EG
bash experiment_iql.sh         # train IQL
bash experiment_td3bc_eg.sh    # train TD3+BC + EG
bash experiment_td3bc.sh       # train TD3+BC
bash experiment_bc_eg.sh       # train BC + EG
bash experiment_bc.sh          # train BC
```

## Citation
If you use or refer to this repository in your research, please cite:

```
@inproceedings{abe2026crossembodimentoffline,
  title     = {Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets},
  author    = {Abe, Haruki and Osa, Takayuki and Mukuta, Yusuke and Harada, Tatsuya},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## Acknowledgements

This work builds on open-source projects from the robotics and reinforcement learning community.

In particular, our implementation was heavily inspired by and developed on top of:

- [RL-X](https://github.com/nico-bohlinger/RL-X) by Nico Bohlinger and Klaus Dorer
- [one_policy_to_run_them_all](https://github.com/nico-bohlinger/one_policy_to_run_them_all) by Nico Bohlinger et al.

We sincerely thank the original authors for making their code and research publicly available.

## License

The code in this repository is licensed under the MIT License. See `LICENSE`.

Some third-party assets included in this repository are provided under their own
licenses and terms.
