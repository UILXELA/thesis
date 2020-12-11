source /home/alex/thesis_ssd/second.pytorch/second/env.sh
python3 fuse_error.py 2>>log_err.txt
mv ./_out1/fused_pc ./_out1/training/velodyne

python3 /home/alex/thesis_ssd/second.pytorch/second/create_data.py kitti_data_prep '/home/alex/thesis_ssd/CARLA_nightly/test_field/_out1'

python3 /home/alex/thesis_ssd/second.pytorch/second/pytorch/train.py evaluate --config_path=/home/alex/thesis_ssd/second.pytorch/second/configs/car.fhd.onestage_test_nightly.config --model_dir=/home/alex/thesis_ssd/second.pytorch/second/817_model/  --measure_time=True --batch_size=1 >>log_err.txt

rm -rf ./_out1/training/velodyne
