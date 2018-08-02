# ITN: Geometry-Aware Scene Text Detection

[Fangfang Wang](https://scholar.google.com/citations?user=T5pyEAoAAAAJ&hl=en)\*, [Liming Zhao](http://www.zhaoliming.net)\*, [Xi Li](http://mypage.zju.edu.cn/xilics), Xinchao Wang, and Dacheng Tao. “[Geometry-Aware Scene Text Detection with Instance Transformation Network](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Geometry-Aware_Scene_Text_CVPR_2018_paper.html).” In CVPR, 2018. (\*authors contributed equally)



```
@InProceedings{Wang_2018_CVPR,
author = {Wang, Fangfang and Zhao, Liming and Li, Xi and Wang, Xinchao and Tao, Dacheng},
title = {Geometry-Aware Scene Text Detection with Instance Transformation Network},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {1381-1389},
year = {2018}
}
```

Contact: 

Fangfang Wang (fangfangliana@gmail.com);

Liming Zhao (http://www.zhaoliming.net/, zlmzju@gmail.com);

Xi Li (http://person.zju.edu.cn/xilics, xilizju@zju.edu.cn)


## Instructions

- Use my [`Caffe`](https://github.com/zlmzju/caffe/tree/itn) for using transformation layer.

- Run `train.sh` in the `experiments` folder to train the model.


## Descriptions

- Use `Caffe` for implementation, please refer to the [Caffe project website](http://caffe.berkeleyvision.org/) for installation.

- The protocal files are located in `experiments/xxx/proto` folder.

- Trained models can be downloaded from [google drive](https://drive.google.com/open?id=1RWmn1KFneV9dhq2ILHCQVM5JSy9zs_zp) or [onedrive](https://1drv.ms/f/s!Ajj620goINVpkBmidAkIVm6GAZmv), and then put them in `experiments/xxx/weight` folder.
