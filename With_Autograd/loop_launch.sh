
#!/bin/bash

export SYSTEM='carla'
export ROOT='./data/carla/images/Dataset_with_image/left_turn_state_space_sampling'
export FNAME='Data_Collection_Compiled_40resize.pd'

# middle of turn
export DATA_PATH_1="${ROOT}/random_noise_driving/${FNAME}"
export DATA_PATH_2="${ROOT}/straight_lane_driving/${FNAME}"
export DATA_PATH_3="${ROOT}/start_of_turn/${FNAME}"
export DATA_PATH_4="${ROOT}/middle_of_turn/${FNAME}"

export RESULTS_PATH='./results-tmp'

# Margins for optimization constraints 
export GAMMA_SAFE=0.05
export GAMMA_UNSAFE=0.05
export GAMMA_DYN=0.01

# Lagrange multipliers (fixed)
export LAMBDA_GRAD=0.01
export LAMBDA_PARAM=0.01

# Robustness
export DELTA_F=0.1
export DELTA_G=0.1

# Output map
#export LIP_CONST_1=0.1
#export LIP_CONST_2=0.1


export LIP_CONST_1=1
export LIP_CONST_2=1


# Training
export NET_DIMS=(32 16)
export N_EPOCHS=2000
export LEARNING_RATE=0.005
export DUAL_STEP_SIZE=0.05
export DUAL_SCHEME='ae'

# For clean data with data aug
export NEIGHBOR_THRESH=0.008
export MIN_N_NEIGHBORS=200

# Additional state sampling
export N_SAMP_UNSAFE=0
export N_SAMP_SAFE=0
export N_SAMP_ALL=0

export CUDA_VISIBLE_DEVICES=0
# Loop through the iterations
for ((i=1; i<=20; i++))
do
    # Create a unique results path for each iteration
    export RESULTS_PATH="./results/results_${i}"
    echo "Iteration: $i"

    # Set the CUDA_VISIBLE_DEVICES if needed
    

    # Run the main file with the desired arguments
    python main.py \
      --system $SYSTEM --data-path $DATA_PATH_1 $DATA_PATH_2 $DATA_PATH_3 $DATA_PATH_4 \
      --results-path $RESULTS_PATH \
      --gamma-safe $GAMMA_SAFE --gamma-unsafe $GAMMA_UNSAFE --gamma-dyn $GAMMA_DYN \
      --lambda-grad $LAMBDA_GRAD --lambda-param $LAMBDA_PARAM \
      --net-dims ${NET_DIMS[@]} --n-epochs $N_EPOCHS \
      --learning-rate $LEARNING_RATE --dual-step-size $DUAL_STEP_SIZE \
      --nbr-thresh $NEIGHBOR_THRESH --min-n-nbrs $MIN_N_NEIGHBORS \
      --n-samp-unsafe $N_SAMP_UNSAFE --n-samp-safe $N_SAMP_SAFE --n-samp-all $N_SAMP_ALL \
      --dual-scheme $DUAL_SCHEME \
      --robust --delta-f $DELTA_F --delta-g $DELTA_G \
      --use-lip-output-term --lip-const-a $LIP_CONST_1 --lip-const-b $LIP_CONST_2

    # Optionally, perform any additional actions with the results, such as copying or moving them
done
