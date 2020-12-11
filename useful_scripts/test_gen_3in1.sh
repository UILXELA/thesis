source ../env2.sh
for i in {1..6}
do
	echo $i

	while :
        do
                if [ $(($(date +%s) % 20)) == 0 ]
                then
                        break
                fi
        done
	
	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/test_gen_bridge.sh &
	sleep 5
	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/change_weather.sh
	sleep 1800
	pkill -9 python
	pkill -9 Carla
	sleep 10

	while :
	do
        	if [ $(($(date +%s) % 20)) == 0 ]
        	then
                	break
        	fi
	done


	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/test_gen_highway.sh &
	sleep 5
	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/change_weather.sh
	sleep 1800
	pkill -9 python
	pkill -9 Carla
        sleep 10

	while :
        do
                if [ $(($(date +%s) % 20)) == 0 ]
                then
                        break
                fi
        done


	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/test_gen_round.sh &
	sleep 5
	bash /home/alex/thesis_ssd/CARLA_nightly/carla-training-data/change_weather.sh
	sleep 1800
        pkill -9 python
	pkill -9 Carla
        sleep 10

	#while :
        #do
        #        if [ $(($(date +%s) % 20)) == 0 ]
        #        then
        #                break
        #        fi
        #done
done
