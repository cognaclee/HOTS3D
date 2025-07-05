# HOTS3D
Hyper-spherical Optimal Transport for Semantic Alignment in Text-to-3D End-to-end Generation
## Introduction
![Full pipeline of our hots3d](assets/pipeline.png)
Our framework for text-guided 3D synthesis comprises three stages. Firstly, we encode the input text prompt onto the hypersphere with a pre-trained CLIP text encoder, obtaining text features. Secondly, the SOT map is induced by the gradient of a convex function that is trained via minimax optimization, and then transfers output text
features to the image feature space. In the third stage, a generator conditioned on the output of the SOT Map was utilized to generate 3D shapes. The SOT map is a plug-
and-play tool for aligning spherical distributions. During the training phase, we only need to optimize the parameters
of the SOT map and other modules remain frozen, significantly reducing the training difficulty. With the SOT map for semantic alignment, our HOTS3D can bypass iterative
optimization during the testing phase, resulting in stronger generalization capability and semantic consistency.
## Citation
If you find our work useful in your research, please consider citing:

```
@misc{li2024hots3dhypersphericaloptimaltransport,
      title={HOTS3D: Hyper-Spherical Optimal Transport for Semantic Alignment of Text-to-3D Generation}, 
      author={Zezeng Li and Weimin Wang and WenHai Li and Na Lei and Xianfeng Gu},
      year={2024},
      eprint={2407.14419},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14419}, 
}
```
## Performences
![qualitative.png](assets/qualitative.png)

## Usage
### Install requirements using following scripts.
```bash
git clone https://github.com/cognaclee/HOTS3D.git
cd HOTS3D

coda create -n HOTS3D python=3.10
conda activate HOTS3D
pip install -r reqirements.txt
```

