# Configuration Guide

This section provides a detailed description of the configuration files used in Trinity-RFT.

## Overview

The configuration of Trinity-RFT is a `yaml` file, which is divided into several parts according to different modules. Below is an example of a configuration file:

```yaml
project: Trinity-RFT
name: tutorial
mode: both
checkpoint_root_dir: /PATH/TO/CHECKPOINT

algorithm:
  # for algorithm related parameters
  ...
model:
  # models used for training
  ...
cluster:
  # number and specifications of cluster nodes
  ...
buffer:
  # data buffer for explorer and trainer
  ...
explorer:
  # rollout models and workflow runners settings
  ...
trainer:
  # parameters related to specific training engines
  ...
synchronizer:
  # model weights synchronization method and interval
  ...
monitor:
  # monitor settings
  ...
data_processor:
  # settings for process the data before training
  ...
```


In the following sections, we will provide a detailed description of each part of the configuration.
Since the RFT process is relatively complex and involves many parameters, this document will focus on those items that require special attention. For other parameters, please refer to the [code](https://github.com/modelscope/Trinity-RFT/blob/main/trinity/common/config.py).


## Global Configs


```yaml
project: Trinity-RFT
name: example
mode: both
checkpoint_root_dir: /PATH/TO/CHECKPOINT
```

- `project`: The name of the project.
- `name`: The name of the experiment.
- `mode`: The running mode of Trinity-RFT, chosen from `both`, `train`, `explore` or `bench`.
  - In `both` mode both trainer and explorer are launched, which is the default mode.
  - In `train` mode only trainer is launched.
  - In `train` mode only explorer is launched.
  - The `bench` mode is used for benchmark evaluation.
- `checkpoint_root_dir`: The root directory to save checkpoints. This directory is the root path of the workspace and can be used to organize the results of multiple experiments. Sepcifically, the checkpoints of this experiment will be saved in `<checkpoint_root_dir>/<project>/<name>/`.

## Algorithm

The algorithm configuration is used to specify the algorithm type and other algorithm parameters.

```yaml
algorithm:
  algorithm_type: grpo
  repeat_times: 1
  gamma: 1.0
  lam: 1.0
```

- `algorithm.algorithm_type`: The type of the algorithm. Support `ppo`, `grpo`, `opmd` and `dpo`.
- `algorithm.repeat_times`: The number of times to repeat each task. Used for GRPO-like algorithm. Default is `1`. In `dpo`, the value of this field will be automatically filled in as `2`.
- `algorithm.gamma`: The discount factor for the value function. Default is `1.0`.
- `algorithm.lam`: The lambda for the generalized advantage estimation. Default is `1.0`.

## Monitor

The monitor is used to log the training process for both explorer and trainer.

```yaml
monitor:
  monitor_type: wandb
```

- `monitor.monitor_type`: The type of the monitor. For now, `MonitorType.WANDB` and `MonitorType.TENSORBOARD` are supported.
  - When using `wandb`, you need to login to your WandB account and set the environment variable (`WANDB_API_KEY`) properly before running the experiment. The generated wandb experiement's project and name are the same as the `project` and `name` in global configs.
  - When using `tensorboard`, the generated file will be saved in `<checkpoint_root_dir>/<project>/<name>/monitor/tensorboard`.


## Model

The `model` configuration specifies the model used for training.

```yaml
model:
  model_path: '/PATH/TO/MODEL/CHECKPOINT/'
  critic_model_path: ''
  max_prompt_tokens: 4096
  max_response_tokens: 16384
```

- `model.model_path`: The checkpoint path of the model to be trained.
- `model.critic_model_path`: The path to the critic model checkpoint. If not set, the `model.critic_model_path` will be set to `model.model_path`.
- `model.max_prompt_tokens`: The maximum number of tokens in the prompt of the model.
- `model.max_response_tokens`: The maximum number of tokens in the response of the model.

## Cluster

The `cluster` configuration specifies the cluster configuration. It includes the number of nodes and the number of GPUs per node.

```yaml
cluster:
  node_num: 1
  gpu_per_node: 8
```

- `cluster.node_num`: The number of nodes in the cluster used for training.
- `cluster.gpu_per_node`: The number of GPUs per node.

## Buffer

The `buffer` configuration specifies the data buffer for the explorer and trainer. This part is relatively complicated but very important. For ease of understanding, we will introduce the `buffer` configs used by explorer and trainer respectively.


```yaml
buffer:
  batch_size: 32
  total_epochs: 100

  explorer_input:
    taskset:
      ...
    eval_tasksets:
      ...

  trainer_input:
    experience_buffer:
      ...
    sft_warmup_dataset:
      ...

  default_workflow_type: 'math_workflow'
  default_reward_fn_type: 'countdown_reward'
```

- `buffer.batch_size`: The number of data item to be sampled from the buffer for training. *Please do not multiply this value by the `algorithm.repeat_times` manually*.
- `buffer.total_epochs`: The total number of epochs to train. This parameter will not take effect when using a buffer with streaming data (e.g., buffers of `queue` type).


### Explorer Input

This part is used to specify the input of the explorer. The explorer need two different set of data, `taskset` and `eval_tasksets`. Below is an example.

```yaml
buffer:
  ...
  explorer_input:
    taskset:
      name: countdown_train
      storage_type: file
      path: /PATH/TO/DATA
      split: train
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'


    eval_tasksets:
    - name: countdown_eval
      storage_type: file
      path: /PATH/TO/DATA
      split: test
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 0.1
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'


```

- `buffer.explorer_input.taskset`: The task dataset to use in explorer for training. For now, we only support one taskset here. In the future, we will support multiple tasksets here.
- `buffer.explorer_input.eval_taskset`: The list of task dataset to use in explorer for evaluation.


The configuration for each task dataset is defined as follows:

- `name`: The name of the task dataset. It needs to be globally unique, and data processing module will use this name to load the dataset in future versions.
- `storage_type`: The storage type of the task dataset. For now, we only support `file`, `queue` and `sql` storage type.
  - `file`: The task dataset is stored in `jsonl`/`parquet` files. The data file organization is required to meet the huggingface standard. *We recommand using this storage type for most cases.*
  - `queue`: The task dataset is stored in a queue. The queue is a simple FIFO queue that stores the task dataset. *Do not use this storage type for task dataset unless you know what you are doing.*
  - `sql`: The task dataset is stored in a SQL database. *This type is unstable and will be optimized in the future versions.*
- `path`: The path to the task dataset.
  - For `file` storage type, the path is the path to the directory that contains the task dataset files.
  - For `queue` storage type, the path is optional. You can back up the data in the queue by specifying a sqlite database path here.
  - For `sql` storage type, the path is the path to the sqlite database file.
- `format`: The format of the task dataset. Only for `file` storage type.
  - `prompt_key`: Specifies which column in the dataset contains the prompt data.
  - `response_key`: Specifies which column in the dataset contains the response data.
- `rollout_args`: The parameters for rollout.
  - `temperature`: The temperature for sampling.
- `default_workflow_type`: The default workflow type for this task dataset. If not specified, use the `buffer.default_workflow_type`
- `default_reward_fn_type`: The default reward funtion type for this task dataset. If not specified, use the `buffer.default_reward_fn_type`.

### Trainer Input

```yaml
buffer:
  ...
  trainer_input:
    experience_buffer:
      name: countdown_buffer
      storage_type: queue
      path: sqlite:///countdown_buffer.db
    sft_warmup_dataset:
      name: warmup_data
      storage_type: file
      path: /PATH/TO/WARMUP_DATA
      format:
        prompt_key: 'question'
        response_key: 'answer'
    sft_warmup_steps: 0
```

- `buffer.trainer_input.experience_buffer`: The experience buffer to use in the trainer.
- `buffer.trainer_input.experience_buffer.name`: The name of the experience buffer. It should be globally unique.
- `buffer.trainer_input.experience_buffer.storage_type`: Similar to the `storage_type` in explorer input dataset, but we only recommend `queue` here. `sql` and `file` will be supported in the future.
- `buffer.trainer_input.sft_warmup_dataset`: The dataset to use for SFT warmup in the trainer. It has the same format as the task dataset in the explorer input. This field is optional, set it only if you want to use SFT warmup.
- `buffer.trainer_input.sft_warmup_steps`: The number of steps to use for SFT warmup in the trainer. If none-zero, `buffer.trainer_input.sft_warmup_dataset` must be set.


## Explorer

The `explorer` configuration is used to configurate the workflow and rollout related functionality.

```yaml
explorer:
  runner_num: 32
  rollout_model:
    engine_type: vllm_async
    engine_num: 1
    tensor_parallel_size: 1
    enable_prefix_caching: false
    dtype: bfloat16
    seed: 42
```
- `runner_num`: The number of worklow runners. We recommand to set it to at least 4 times of the number of rollout models to improve the throughput, but at the same time do not exceed the `explorer.batch_size`.
- `explorer.rollout_model.engine_num`: The number of rollout engines. Default is `1`.
- `explorer.rollout_model.engine_type`: The type of the engine. support `vllm_async` and `vllm`. Default is `vllm_async`. We recommand using `vllm_async` here, and `vllm` is may be removed in the future.
- `explorer.rollout_model.tensor_parallel_size`: The tensor parallel size used in vLLM. Default is `1`.
- `explorer.rollout_model.enable_prefix_caching`: Whether to enable prefix caching. Default is `False`.
- `explorer.rollout_model.dtype`: The data type used in vLLM. Default is `bfloat16`.
- `explorer.rollout_model.seed`: The seed used in vLLM. Default is `42`.
- `explorer.rollout_model.use_v1`: Whether to use v1 of vLLM. Default is `True`. We will remove this item after the v1 engine is stable and use v1 by default.
- `explorer.rollout_model.enable_openai_api`: Whether to enable OpenAI API. Default is `False`.
- `explorer.rollout_model.enable_thinking`: For Qwen3, whether to enable thinking. Default is `False`.
- `explorer.rollout_model.chat_template`: To override the default chat template of the model. If not specified, the default chat template will be used. Default is `None`.

## Synchronizer

```yaml
synchronizer:
  sync_method: 'nccl'
  sync_interval: 10
  sync_timeout: 1200
```

- `synchronizer.sync_method`: The synchronization method between `trainer` and `explorer`.
Support `nccl` and `checkpoint`, `nccl` represents that model weights in `explorer` will be synchronized from `trainer` through `nccl`,
`checkpoint` represents that `explorer` will load the newest checkpoints saved by `trainer` then update its model weights. Default is `nccl`.
- `synchronizer.sync_interval`: The interval steps between two synchronizations. Default is `10`. It should be set manually.
- `synchronizer.sync_timeout`: The timeout of the synchronization. Default is `1200`.

## Trainer

```yaml
trainer:
  trainer_type: 'verl'
  trainer_config_path: 'examples/ppo_countdown/train_countdown.yaml'
  save_interval: 100
```

- `trainer.trainer_type`: The backend of the trainer, Only `verl` is supported.
- `trainer.trainer_config_path`: The path to the trainer configuration file. It must be set manually.
- `trainer.save_interval`: The interval steps between two checkpoints. Default is `100`.

## Data Processing

<!-- The `data` configuration specifies the data used for training. It includes the total number of epochs, the batch size, the path to the dataset, the default workflow type, the default reward function type, and the format configuration. -->

```yaml
data_processor:
  source_data_path: '/PATH/TO/DATASET'
  load_kwargs:
    split: 'train'  # only need the train split
  format:
    prompt_key: 'question'
    response_key: 'answer'

  # cleaner related
  dj_config_path: 'tests/test_configs/active_iterator_test_dj_cfg.yaml'
  clean_strategy: 'iterative'
  # db related
  db_url: 'postgresql://{username}@localhost:5432/{db_name}'
```

- `data.source_data_path`: The path to the source dataset.
- `data.load_kwargs`: The kwargs used in `datasets.load_dataset`.
- `data.format`: The format of the source dataset. It includes `prompt_key` and `response_key`.
- `data.dj_config_path`: The path to the Data-Juicer configuration.
- `data.clean_strategy`: The cleaning strategy used for `DataCleaner`, which iteratively cleans dataset until targets are met.
- `data.db_url`: The URL of the database.

### veRL Trainer Configuration

Here we mainly introduce the parameters that can be set in veRL. For the specific meaning of the parameters, please refer to the official document of [veRL](https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/docs/examples/config.rst).

```yaml
data:
  tokenizer: null
  train_files: train_example.parquet
  val_files: test_example.parquet
  prompt_key: prompt
  max_prompt_length: 256
  max_response_length: 1024
  train_batch_size: 256
  val_batch_size: null
  return_raw_input_ids: False  # This should be set to true when the tokenizer between policy and rm differs
  return_raw_chat: False
  shuffle: True
  filter_overlong_prompts: False # for large-scale dataset, filtering overlong prompts could be timeconsuming. You should disable this and set `truncation='left'
  truncation: error
  image_key: images

actor_rollout_ref:
  hybrid_engine: True
  model:
    path: /PATH/TO/MODEL/CHECKPOINT/
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: True
    use_remove_padding: True
  actor:
    strategy: fsdp  # This is for backward-compatibility
    ppo_mini_batch_size: 128
    # ppo_micro_batch_size: 8 # will be deprecated, use ppo_micro_batch_size_per_gpu
    ppo_micro_batch_size_per_gpu: 4
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
    grad_clip: 1.0
    clip_ratio: 0.2
    entropy_coeff: 0.001
    use_kl_loss: False # True for GRPO
    kl_loss_coef: 0.001 # for grpo
    kl_loss_type: low_var_kl # for grpo
    ppo_epochs: 1
    shuffle: False
    ulysses_sequence_parallel_size: 1 # sp size
    checkpoint:
      contents: ['model', 'hf_model', 'optimizer', 'extra']  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
    optim:
      lr: 1e-6
      lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
      # min_lr_ratio: null   # only useful for warmup with cosine
      warmup_style: constant  # select from constant/cosine
      total_training_steps: -1  # must be override by program
    fsdp_config:
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1
    # --- below: opmd ---
    tau: 0.000  # strength of regularization w.r.t. old / ref policy
    opmd_baseline: mean  # mean / logavgexp, applicable to opmd
    use_uid: False  # True / False, applicable to pairwise_opmd
  ref:
    fsdp_config:
      param_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
    # log_prob_micro_batch_size: 4 # will be deprecated, use log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 8
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
  rollout:
    name: vllm
    temperature: 1.0
    top_k: -1 # 0 for hf rollout, -1 for vllm rollout
    top_p: 1
    use_fire_sampling: False # https://arxiv.org/abs/2410.21236
    prompt_length: ${data.max_prompt_length}  # not use for opensource
    response_length: ${data.max_response_length}
    # for vllm rollout
    dtype: bfloat16 # should align with FSDP
    gpu_memory_utilization: 0.4
    ignore_eos: False
    enforce_eager: True
    free_cache_engine: True
    load_format: dummy_dtensor
    tensor_model_parallel_size: 2
    max_num_batched_tokens: 8192
    max_model_len: null
    max_num_seqs: 1024
    # log_prob_micro_batch_size: 8 # will be deprecated, use log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    disable_log_stats: True
    enable_chunked_prefill: True # could get higher throughput
    # for hf rollout
    do_sample: True
    # number of responses (i.e. num sample times)
    n: 1 # > 1 for grpo

critic:
  strategy: fsdp
  optim:
    lr: 1e-5
    lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
    # min_lr_ratio: null   # only useful for warmup with cosine
    warmup_style: constant  # select from constant/cosine
    total_training_steps: -1  # must be override by program
  model:
    path: /PATH/TO/MODEL/CHECKPOINT/
    tokenizer_path: ${actor_rollout_ref.model.path}
    override_config: { }
    external_lib: ${actor_rollout_ref.model.external_lib}
    enable_gradient_checkpointing: True
    use_remove_padding: False
    fsdp_config:
      param_offload: False
      optimizer_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      fsdp_size: -1
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
  # ppo_micro_batch_size: 8 # will be deprecated, use ppo_micro_batch_size_per_gpu
  ppo_micro_batch_size_per_gpu: 8
  forward_micro_batch_size_per_gpu: ${critic.ppo_micro_batch_size_per_gpu}
  use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
  ppo_max_token_len_per_gpu: 32768 # (${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}) * 2
  forward_max_token_len_per_gpu: ${critic.ppo_max_token_len_per_gpu}
  ulysses_sequence_parallel_size: 1 # sp size
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  shuffle: ${actor_rollout_ref.actor.shuffle}
  grad_clip: 1.0
  cliprange_value: 0.5

reward_model:
  enable: False
  strategy: fsdp
  model:
    input_tokenizer: ${actor_rollout_ref.model.path}  # set this to null if the chat template is identical
    path: ~/models/FsfairX-LLaMA3-RM-v0.1
    external_lib: ${actor_rollout_ref.model.external_lib}
    use_remove_padding: False
    fsdp_config:
      min_num_params: 0
      param_offload: False
      fsdp_size: -1
  # micro_batch_size: null # will be deprecated, use micro_batch_size_per_gpu
  # micro_batch_size_per_gpu: 2 # set a number
  # max_length: null
  ulysses_sequence_parallel_size: 1 # sp size
  use_dynamic_bsz: ${critic.use_dynamic_bsz}
  forward_max_token_len_per_gpu: ${critic.forward_max_token_len_per_gpu}
  reward_manager: tinyzero

custom_reward_function:
  path: null
  name: compute_score

algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  norm_adv_by_std_in_grpo: True
  use_kl_in_reward: False
  kl_penalty: kl  # how to estimate kl divergence
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
    horizon: 10000
    target_kl: 0.1

trainer:
  balance_batch: True
  total_epochs: 15
  # total_training_steps: null
  project_name: TinyZero
  experiment_name: trinity-qwen2.5-1.5b
  logger: [ 'wandb' ]
  val_generations_to_log_to_wandb: 0
  nnodes: 1
  n_gpus_per_node: 2
  save_freq: 100
  # auto: find the last ckpt to resume. If can't find, start from scratch
  resume_mode: auto # or auto or resume_path if
  resume_from_path: ""
  test_freq: 100
  critic_warmup: 0
  default_hdfs_dir: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  val_before_train: False
  max_actor_ckpt_to_keep: 5
  max_critic_ckpt_to_keep: 5
```


- `actor_rollout_ref.model.enable_gradient_checkpointing`: Whether to enable gradient checkpointing, which will reduce GPU memory usage.
- `actor_rollout_ref.model.use_remove_padding`: Whether to remove pad tokens, which will reduce training time.
- `actor_rollout_ref.actor.use_dynamic_bsz`: Whether to reorganize the batch data, specifically to splice the shorter data to reduce the batch size in the actual training process.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Batch size for one GPU in one forward pass.
- `actor_rollout_ref.actor.grad_clip`: Gradient clip for actor model training.
- `actor_rollout_ref.actor.clip_ratio`: Used for compute policy loss.
- `actor_rollout_ref.actor.entropy_coeff`: Used for compute policy loss.
- `actor_rollout_ref.actor.use_kl_loss`: Whether to enable kl loss.
- `actor_rollout_ref.actor.kl_loss_coef`: The coefficient of kl loss.
- `actor_rollout_ref.actor.kl_loss_type`: How to compute kl loss, optional value is `kl`, `abs`, `mse` or `low_var_kl`.
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`: Ulysses sequence parallel size.
- `actor_rollout_ref.actor.tau`: strength of regularization w.r.t. old / ref policy.
- `actor_rollout_ref.actor.opmd_baseline`: mean / logavgexp, applicable to opmd.
- `actor_rollout_ref.actor.use_uid`: True / False, applicable to pairwise_opmd.
- `actor_rollout_ref.actor.optim.lr`: Learning rate for actor model.
- `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`: Ratio of warmup steps for learning rate.
- `actor_rollout_ref.actor.optim.warmup_style`: Warmup style for learning rate.
- `actor_rollout_ref.actor.optim.total_training_steps`: Total training steps for actor model.
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: Batch size for one GPU in one reference model forward pass.

- `critic.model.enable_gradient_checkpointing`: Whether to enable gradient checkpointing, which will reduce GPU memory usage.
- `critic.model.use_remove_padding`: Whether to remove pad tokens, which will reduce training time.
- `critic.optim.lr`: Learning rate for critic model.
- `critic.optim.lr_warmup_steps_ratio`: Ratio of warmup steps for learning rate.
- `critic.optim.warmup_style`: Warmup style for learning rate.
- `critic.optim.total_training_steps`: Total training steps for critic model.
- `critic.ppo_micro_batch_size_per_gpu`: Batch size for one GPU in one critic model forward pass.
- `critic.ulysses_sequence_parallel_size`: Ulysses sequence parallel size.
- `critic.grad_clip`: Gradient clip for critic model training.
- `critic.cliprange_value`: Used for compute value loss.

- `algorithm`: Training algorithm settings.

- `trainer.balance_batch`: Whether to balance batch size between GPUs during training.
- `trainer.resume_mode`: Resume mode for training. Support `disable`, `auto` and `resume_path`.
- `trainer.resume_from_path`: Path to resume from.
- `trainer.critic_warmup`: The number of steps to train the critic model before actual policy learning.
- `trainer.default_hdfs_dir`: Default HDFS directory for saving checkpoints.
- `trainer.remove_previous_ckpt_in_save`: Whether to remove previous checkpoints in save.
- `trainer.del_local_ckpt_after_load`: Whether to delete local checkpoints after loading.
- `trainer.max_actor_ckpt_to_keep`: Maximum number of actor checkpoints to keep.
- `trainer.max_critic_ckpt_to_keep`: Maximum number of critic checkpoints to keep.
