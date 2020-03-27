# Conditional WaveGAN

This is the official implementation of [Conditional WaveGAN](https://arxiv.org/abs/1809.10636).

In this paper, we developed Conditional WaveGAN to synthesize speech/audio samples that are conditioned on class labels. The thus synthesized raw audio is used for improving the baseline ASR system.

## Motivation

Generative models are successfully used for image synthesis in the recent years. But when it comes to other modalities like audio, text, and etc, little progress has been made. Recent works focus on generating audio from a generative model in an unsupervised setting. We explore the possibility of using generative models **conditioned on class labels**.

## Methods

<img src="examples/concat.jpeg"/>

<img src="examples/bias.jpeg"/>

<img src="examples/generation.jpeg"/>

## Usage

Training can be done in both GPU and TPU settings. Both versions of code implement bias scaling method.

### Prerequisites

* Tensorflow 1.x.x
* Python 2.x, 3.x
* tqdm


### Datasets

1. Techsorflow Challenge [Raw SC09](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
2. [TFRecord SC09](https://drive.google.com/file/d/1yX5iFZ9sqaD4_9-OhmsISZ1PkDnGjHse/view?usp=sharing)

Data must assume the form of `tf.Data.TFRecord`. The label data must be in one hot encoded for concatenation based conditioning, whereas it must be simple integers for bias based conditioning. Thus, the code to make the TFRecord differs by the type of conditioning.

```
python3 make_tfrecord_int.py \
	../sc09/train \
	../sc09_tf \
	--name train --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1
```

### Training in GPU

To begin or resume training

```shell
python gpu/train_wavegan.py train ./gpu/train \
	--data_dir ./data/customdataset
```

To log and monitor training

```shell
# saves checkpoints every 60 minutes
python gpu/backup.py ./gpu/train 60

# generates preview audio files every time a checkpoint is saved
python gpu/train_wavegan.py preview ./gpu/train
```

Note that the training directory for train, backup, and preview must be identical.

### Training in TPU

Setting up TPU is explained [here](https://medium.com/@cylee_80935/how-to-use-google-cloud-tpus-177c3a025067).

To begin or resume training

```
python tpu/tpu_main.py
```

Create a bucket for backup checkpoints and name it `[CKPT_BUCKET_NAME]-backup`. To save the checkpoints every specified minutes while training

```
# save checkpoints every 60 minutes
python tpu/backup.py gs://ckpt 60
```

To generate 20 preview audio samples with two per class

```
python tpu/preview.py
```

### Synthesized audio samples (Demo)

https://colab.research.google.com/drive/1VRyNJQBgiFF-Gi9qlZkOhiBE-KkUaHjw

### References

* Donahue, Chris, Julian McAuley, and Miller Puckette. "Synthesizing Audio with Generative Adversarial Networks." arXiv preprint arXiv:1802.04208 (2018). [paper](https://arxiv.org/abs/1802.04208)
* Shen, Jonathan, et al. "Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions." arXiv preprint arXiv:1712.05884 (2017). [paper](https://arxiv.org/pdf/1712.05884.pdf)
* Perez, Anthony, Chris Proctor, and Archa Jain. Style transfer for prosodic speech. Tech. Rep., Stanford University, 2017. [paper](http://web.stanford.edu/class/cs224s/reports/Anthony_Perez.pdf)
* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014. [paper](https://arxiv.org/pdf/1406.2661.pdf)
* Salimans, Tim, et al. "Improved techniques for training gans." Advances in Neural Information Processing Systems. 2016. [paper](https://arxiv.org/pdf/1606.03498.pdf)
* Grinstein, Eric, et al. "Audio style transfer." arXiv preprint arXiv:1710.11385 (2017). [paper](https://arxiv.org/abs/1710.11385)
* Pascual, Santiago, Antonio Bonafonte, and Joan Serra. "SEGAN: Speech enhancement generative adversarial network." arXiv preprint arXiv:1703.09452 (2017). [paper](https://arxiv.org/pdf/1703.09452.pdf)
* Yongcheng Jing, Yezhou Yang, Zunlei Feng, Jingwen Ye, Yizhou Yu, Mingli Song. "Neural Style Transfer: A Review" 	arXiv:1705.04058 (2017) [paper](https://arxiv.org/abs/1705.04058v6)
* Van Den Oord, Aäron, et al. "Wavenet: A generative model for raw audio." CoRR abs/1609.03499 (2016). [paper](https://arxiv.org/abs/1609.03499)
* Glow: Generative Flow with Invertible 1×1 Convolutions [paper](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf)
* Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014. [paper](https://arxiv.org/abs/1406.5298)
* Van Den Oord, Aäron, et al. "Wavenet: A generative model for raw audio." CoRR abs/1609.03499 (2016). [paper](https://arxiv.org/abs/1609.03499)


## Authors

* **Anoop Toffy** - *IIIT Bangalore* - [Personal Website](https://www.anooptoffy.com)
* **Chae Young Lee** - *Hankuk Academy of Foreign Studies* - [Homepage](https://github.com/acheketa)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Credits

* We used our baseline mode from waveGAN paper by Chris Donahue et al. (2018)

```
@article{donahue2018synthesizing,
  title={Synthesizing Audio with Generative Adversarial Networks},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  journal={arXiv preprint arXiv:1802.04208},
  year={2018}
}
```

* TPU Implementations are based on the [DCGAN](https://github.com/tensorflow/tpu/tree/master/models/experimental/dcgan) implemenatation released by Tensorflow Hub. [link](https://github.com/tensorflow/tpu)

## Acknowledgments

* Dr. Gue Jun Jung, Speech Recognition Tech, SK Telecom
* Dr. Woo-Jin Han, Netmarble IGS
* Google Mentors
* Tensorflow Korea
* Google

This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/).
