#!/bin/bash
# Script to run iperf3 benchmark 10 times and collect results

POD1="nccl-golden-custom-6f5c85f77b-xr4fl"
SERVER_IP="10.30.0.2"

echo "Running iperf3 10 times..."

for i in {1..10}; do
    echo "---------------------"
    echo "Run $i/10"
    
    # Run iperf3 and capture output
    kubectl exec $POD1 -- iperf3 -c $SERVER_IP --bidir -P 8 > run_iperf_$i.log 2>&1
    
    # Extract result (both TX and RX)
    tx_bw=$(grep "\[SUM\]\[TX-C\]" run_iperf_$i.log | grep "receiver" | awk '{print $6}')
    rx_bw=$(grep "\[SUM\]\[RX-C\]" run_iperf_$i.log | grep "receiver" | awk '{print $6}')
    
    echo "Run $i TX: $tx_bw Gbits/sec, RX: $rx_bw Gbits/sec"
done

echo "---------------------"
echo "All runs complete. Log files saved as run_iperf_1.log to run_iperf_10.log"
