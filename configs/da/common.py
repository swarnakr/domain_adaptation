'''*************************************************************************
	> File Name: common.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 01:46:28 PM EDT
 ************************************************************************'''
class Config:
    #Model
    backbone = 'resnet' #backbone name ['resnet', 'xception', 'drn', 'mobilenet']
    out_stride = 16 #network output stride

    #Data
#    all_dataset = ['Paris', 'Khartoum', 'Shanghai', 'Vegas','ShanghaiVegas']
#    dataset = 'ShanghaiVegas'
    all_dataset = [ 'Shanghai','Paris']
    dataset = 'Shanghai'
    train_num_workers = 4
    val_num_workers = 2
    gt_root = '/usr/xtmp/satellite/spacenet/'

    train = 1
    test_transformed = 0;
    mean_calc=0
    #params to change
    #tr.Normalize(mean=(0.3441, 0.3809, 0.4014), std=(0.1883, 0.2039, 0.2119)), #regular         
    #tr.Normalize(mean=(0.2702353,  0.3288833,  0.35751825), std=(0.18664147, 0.20198943, 0.2105174)), # style
    #mean_std = ((0.32489884, 0.36593837, 0.39595861),(0.20834632, 0.19246615, 0.18791017)) style_clean
    
    if train:
        mean_std = ((0.3441, 0.3809, 0.4014), (0.1883, 0.2039, 0.2119))
        img_root = '/usr/xtmp/satellite/spacenet/'
        if_augment = 0 #1
        repeat_count = 0 #1        
        transform_sample = 1

        if 0:
            #mean_std = ((0.29290962, 0.33944466, 0.37961883),(0.14136458, 0.12535075, 0.11518841)) #v2 no transform
            #mean_std = ((0.2415185,  0.27953407, 0.31290946),(0.16626957, 0.16843738, 0.17422392)) #v2 with transform
        
            img_root = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/nst/save_files/style_transformed_v2_Shanghai_to_target'
            if_augment = 1
            repeat_count = 5
            transform_sample = 1

    else:
        if test_transformed:
            mean_std = ((0.2415185,  0.27953407, 0.31290946),(0.16626957, 0.16843738, 0.17422392)) #v2 with transf
            #mean_std = ((0.2702353,  0.3288833,  0.35751825), (0.18664147, 0.20198943, 0.2105174))
            img_root = '/usr/xtmp/satellite/spacenet/'
            if_augment = 0
            repeat_count = 0
            transform_sample = 1

        else:
            mean_std = ((0.3441, 0.3809, 0.4014), (0.1883, 0.2039, 0.2119)) #((0.3292, 0.4126, 0.4497), (0.1376, 0.1354, 0.1356))
            img_root = '/usr/xtmp/satellite/spacenet/' #'/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/nst/save_files/style_transformed_Vegas_to_source'
            if_augment = 0 #1
            repeat_count = 0 #1        
            transform_sample = 1

    if not train and mean_calc:
        mean_std = ((0, 0, 0), (1, 1, 1))
        img_root = '/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/nst/save_files/style_transformed_v2_Shanghai_to_target' #'/usr/xtmp/satellite/spacenet/'
        if_augment = 1 #1
        repeat_count = 5 #not used
        transform_sample = 1

    #'/home/home1/swarnakr/main/DomainAdaptation/DA_viaBatchNorm/nst/save_files/style_transformed_Vegas_to_source/' #Shanghai_to_target' #'/usr/xtmp/satellite/spacenet/' Shanghai_to_target'
    #Train
    
    batch_size = 16
    freeze_bn = False
    sync_bn = False
    loss = 'ce' #['ce', 'focal']
    epochs = 100000
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    lr_scheduler = 'cos'
    lr_step = 5
    warmup_epochs = 10

'''
    MEAN = {

    "Shanghai": [0.3292, 0.4126, 0.4497],

    "Vegas": [0.2882, 0.3313, 0.3707],

    "Khartoum": [0.4233, 0.4680, 0.4928],

    "Paris": [0.1737, 0.2573, 0.3424]

}

STD = {

    "Shanghai": [0.1376, 0.1354, 0.1356],

    "Vegas": [0.1597, 0.1472, 0.1371],

    "Khartoum": [0.2018, 0.2139, 0.2208],

    "Paris": [0.1079, 0.1221, 0.1357]

} '''
    
config = Config()
