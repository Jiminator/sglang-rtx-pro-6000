#!/bin/bash
# Script to run NCCL benchmark 10 times and collect results

POD0="nccl-golden-custom-6f5c85f77b-mc9vx"
POD1="nccl-golden-custom-6f5c85f77b-xr4fl"
MASTER_ADDR="10.30.0.2"
MASTER_PORT="29500"

echo "Running benchmark 10 times..."

for i in {1..10}; do
    echo "---------------------"
    echo "Run $i/10"
    
    # Run Rank 0 and capture output
    kubectl exec $POD0 -- bash -c "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=2 RANK=0 python3 /root/dist_test.py" > run_$i.log 2>&1 &
    PID0=$!
    
    # Run Rank 1 (output not needed for results)
    kubectl exec $POD1 -- bash -c "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=2 RANK=1 python3 /root/dist_test.py" > /dev/null 2>&1 &
    PID1=$!
    
    # Wait for both to finish
    wait $PID0 $PID1
    
    # Extract result
    bw=$(grep "Measured Aggregate AlgBW" run_$i.log | awk '{print $4}')
    echo "Run $i Bandwidth: $bw GB/s"
done

echo "---------------------"
echo "All runs complete. Log files saved as run_1.log to run_10.log"
