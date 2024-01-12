# Mitigating Label Noise through Data Ambiguation

This repository contains an implementation of [Mitigating Label Noise through Data Ambiguation](http://arxiv.org/abs/2305.13764
) to be presented at AAAI-24. Please cite it as follows:

```
@misc{lienen2023mitigating,
      title={Mitigating Label Noise through Data Ambiguation}, 
      author={Julian Lienen and Eyke Hüllermeier},
      year={2023},
      eprint={2305.13764},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Requirements

To install all required packages, you need to run
```
pip install -r requirements.txt
```

The code has been tested using Python 3.9 on Ubuntu 2*.* systems. We trained our models on machines with Nvidia GPUs (we tested CUDA 10.1, 11.1 and 11.6). Furthermore, we recommend to use [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) to get a clean Python environment for the execution without any dependency problems.

As a required prerequisite, the `config.ini` needs to be populated with parameters to set the output directory (`BASE_PATH`), a directory for temporary artifacts (`TMP_PATH`) and an output directory for plots (`PLOT_DIR`). 

## Datasets

All datasets except for CIFAR-10(0)N, WebVision and Clothing1M are downloaded automatically. Webvision is available [here](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html), whereas access to Clothing1M has to be [explicitly granted](https://github.com/Cysu/noisy_label) by the owner. CIFAR-10(0)is available [here](https://github.com/UCSC-REAL/cifar-10-100n). All data needs to be stored in the specified `--data_dir` given as parameter to the training script (see next section).

## Training and Evaluation

For the training and evaluation, you have to call the following function (e.g., for CIFAR-10 with 25 % symmetric synthetic noise for our loss):

```
CUDA_VISIBLE_DEVICES=<the numeric ID(s) of your CUDA device(s)> python train.py --dataset=cifar10  --model resnet34 --seed 0 --loss RDA --adaptive_lrvar2 --adaptive_lrvar2_start_beta 0.75 --lrvar2_beta 0.6 --adaptive_lrvar2_type cosine --lr 0.02 --decay_type cosine --label_noise 0.25
```

`--help` allows for printing out all parameter options. All results presented in the paper were computed based on the training scripts `train.py`.

## License

Our code uses the Apache 2.0 License, which we attached as `LICENSE` file in this repository. 

Feel free to re-use our code. We would be happy to see our ideas put into practice.
