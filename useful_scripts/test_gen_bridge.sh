source ../env2.sh
numactl -C 0 python3 test1_bridge.py -w 0 -n 50 &
sleep 5
numactl -C 1 python3 test2_bridge.py -w 0 -n 0 >>log2 &
#sleep 1
numactl -C 2 python3 test3_bridge.py -w 0 -n 0 >> log3 &
#sleep 1
numactl -C 3 python3 test4_bridge.py -w 0 -n 0 >> log4 &
