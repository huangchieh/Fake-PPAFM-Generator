# Fake-PPAFM-Generator
![](https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2024-05-27-16-34-APrU9x.jpg)

Fake-PPAFM-Generator turns AFM to PPAFM style, which have the potential use in preprocessing experimental images before feeding into the structure discovery machine learning models. This fake PPAFM generator is the inverse generator obtained from CycleGAN training. Using this inverse generator as the preprocessing step makes the structure discovery machine leaning model independent. Therefore, when only need to using PPAFM, the simulation AFM, to train our model without considering the real experimental AFM. 

## Usage
Place the experimental AFM images in folder `image_input/testB`, then run the bash script `0_run.sh` on a GPU node. Then the fake PPAFM images would be generated in folder `image_output`. 


## Results
![](https://cdn.jsdelivr.net/gh/HuangJiaLian/DataBase0@master/uPic/2024-09-30-12-08-ReverseTranslation.png)


## Notes
PS: Rename the name of trained model manually, if you want the model that translate from domain B to A you need to do:

```
latest_net_G_B.pth -> latest_net_G.pth
```

More info can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/233)
