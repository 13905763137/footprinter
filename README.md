# footprinter
Semantic segmentation for detection of buildings in Satellite images

## running the training code

In order to run the code:

```
python train.py -npix (input shape for data preprocessing)
                -lr (learning rate)
                -BS (batch size) 
                -epochs (number of epochs) 
                -MPATH (path for saving the mode) 
                -DPATH (path to the data directory) 
                -model (unet model to use; options: baseline & mobilnet)

```

## Data 

The data directory needs to be organized in the following way: 

```
 data├──train├── images ├── pre
             |          ├── post
             |
             ├── targets├── pre
                        ├── post
                        
     ├──test ├── images ├── pre
             |          ├── post
             |
             ├── targets├── pre
                        ├── post
                        
     ├──valid├── images ├── pre
             |          ├── post
             |
             ├── targets├── pre
                        ├── post
