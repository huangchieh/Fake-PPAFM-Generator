# And example of how to run the style transfer code

for lambda in 10 20 30 40 50 100
do
	# A: PPAFM, B: AFM
	# If there is no out folder, create one
	if [ ! -d image_output/fakeA_lbd_${lambda} ]; then
		mkdir -p image_output/fakeA_lbd_${lambda}
	fi

	# Start style transfer
	python main.py --dataroot image_input --name HyperTest-resnet_6blocks-2-16-${lambda}-0.5 \
	       --model test --netG resnet_6blocks  --ngf 16   --no_dropout --input_nc 1 \
	       --output_nc 1  --checkpoints_dir trained_models \
	       --results_dir image_output/fakeA_lbd_${lambda} \
	       --load_size 192 --crop_size 192\ 
done

