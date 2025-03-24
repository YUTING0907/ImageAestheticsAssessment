# Image-Aesthetics-Assessment
This repo contains the official implementation of the ACMMM 2023 paper:

EAT: An Enhancer for Aesthetics-Oriented Transformers
Shuai He, Anlong Ming, Shuntian Zheng, Haobin Zhong, Huadong Ma
Beijing University of Posts and Telecommunications

[国内的小伙伴请看更详细的中文说明]This repo contains the official implementation of the ACMMM 2023 paper.

### EAT  
Background：Most Transformer-based models tend to generate large prediction errors for background-sensitive images. Therefore, Transformer-based models \textbf{\textit{have not comprehensively surpassed CNN models on IAA tasks yet}}, to our knowledge. However, a lack of the attention to a background is inconsistent with the original intention of a photographic work, e.g., hierarchical compositions are usually formed with the deliberate consideration of background regions. Moreover, the superfluous attention in Transformers usually leads to unnecessarily computational cost and slow convergence on IAA tasks and may even result in overfitting on small IAA datasets. image
EAT: To guide the IAA model to locate more reasonable regions, we present an Enhancer for Aesthetics-Oriented Transformers (EAT) based on the deformable attention, which is able to learn where to locate interest points and how to refine attention by means of offsets for IAA. image image
Performance
Sota on the AVA, TAD66K, FLICKR-AES datasets image image image

* Environment Installation
pandas==0.22.0
nni==1.8
requests==2.18.4
torchvision==0.8.2+cu101
numpy==1.13.3
scipy==0.19.1
tqdm==4.43.0
torch==1.7.1+cu101
scikit_learn==1.0.2
tensorboardX==2.5

### How to Run&Check the Code
download weights from: https://drive.google.com/drive/folders/1UpLYGLU5omztVsIWkRPFTVKAOVe_4p3K?usp=sharing，the weights of pre-train weight dat_base_in1k_224.pth：https://pan.baidu.com/s/1kzXIp8V-QRSLOyRNMA-nUw?pwd=8888 code：8888
download datasets from their official website
run main_nni.py

If you find our work is useful, pleaes cite our paper:
@article{heeat,
  title={EAT: An Enhancer for Aesthetics-Oriented Transformers},
  author={Shuai He, Anlong Ming, Shuntian Zheng, Haobin Zhong, Huadong Ma},
  journal={ACMMM},
  year={2023},
}
