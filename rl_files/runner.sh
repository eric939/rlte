#!/bin/bash
# need to run this script from the root directory of the project !!

echo "starting the script now" 

# optional debug mode: call script with "--debug" to run shorter configs
DEBUG_MODE=false
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_MODE=true
  shift
  echo "debug mode enabled"
fi  

PROJECT_ROOT="${PWD}"
echo $PROJECT_ROOT
PYTHON_SCRIPT="$PROJECT_ROOT/rl_files/actor_critic.py"
echo $PYTHON_SCRIPT

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT"
  exit 1
fi

declare -a ARGS=(
  "noise 20"
  "noise 60"
  "flow 20"
  "flow 60"
  "strategic 20"
  "strategic 60"
)

NUM_STEPS=100
NUM_ENVS=128
NUM_ITERATIONS=400
NUM_EVALUATION_EPISODES=10000

if [[ "$DEBUG_MODE" == true ]]; then
  NUM_STEPS=20
  NUM_ENVS=2
  NUM_ITERATIONS=4
  NUM_EVALUATION_EPISODES=10
fi
TIMESTEPS=$((NUM_ITERATIONS * $NUM_ENVS * $NUM_STEPS))
echo "time steps: "
echo $TIMESTEPS
echo "num steps: "
echo $NUM_STEPS
echo "num envs: "
echo $NUM_ENVS
echo "num iterations: "
echo $NUM_ITERATIONS
echo "num evaluation episodes: "
echo $NUM_EVALUATION_EPISODES


# tag="long_horizon"
# echo "Tag of experiment is set to: $tag"

# redo drift experiments 
# for drop_feature in "None"; 
# do
# for exp_name in "log_normal";
# for exp_name in "log_normal_learn_std"; 
# for drop_feature in "None" "order_info" "volume" "drift"; 
# for drop_feature in "order_info" "volume" "drift"; 
# exp_name="log_normal"
exp_name="dirichlet"
drop_feature="None"
for exp_name in "log_normal" "dirichlet";
do
for args in "${ARGS[@]}"; 
do
  set -- $args 
  ARG1=$1
  ARG2=$2

  echo "#####"
  echo "#####" 
  echo "STARTING A RUN"  
  echo "Running experiment: $exp_name with $ARG1 $ARG2"
  echo "Dropping feature: $drop_feature"
  if [[ -n "$tag" ]]; then
    echo "Tag of experiment is set to: $tag"
    if [[ $ARG2 -eq 60 ]]; then
      ARG2=120
      echo "Running experiment for lot size 120 due to long horizon tag"
    fi
    if [[ $ARG2 -eq 20 ]]; then
      ARG2=40
      echo "Running experiment for lot size 40 due to long horizon tag"
    fi
    # just testing long horizon not general tags here 
    python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$((TIMESTEPS))" --num_envs "$((NUM_ENVS))" --num_steps "$((NUM_STEPS))" --n_eval_episodes "$((NUM_EVALUATION_EPISODES))" --exp_name "$exp_name" --time_delta 30 --terminal_time 300 --tag "$tag" --drop_feature "$drop_feature"
  else
    python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$((TIMESTEPS))" --num_envs "$((NUM_ENVS))" --num_steps "$((NUM_STEPS))" --n_eval_episodes "$((NUM_EVALUATION_EPISODES))" --exp_name "$exp_name" --drop_feature "$drop_feature"
  fi
done
done 

echo "All processes completed."