
# RestorMixer: An Efficient and Mixed Heterogeneous Model for Image Restoration 
[![arXiv](https://img.shields.io/badge/arXiv-2504.10967-<COLOR>.svg)](https://arxiv.org/abs/2504.10967)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ClimBin/RestorMixer)

RestorMixer is a novel general-purpose image restoration (IR) model that combines the strengths of multiple architectural paradigms—**Convolutional Neural Networks (CNNs)**, **Transformers**, and **Mamba**—to achieve state-of-the-art performance with high inference efficiency. This project is the official implementation of the paper, *“An Efficient and Mixed Heterogeneous Model for Image Restoration”*.

## 🌟 Why RestorMixer?

Image restoration is a critical task for recovering high-quality images from degraded ones, but traditional methods are often task-specific and costly to develop. While recent general-purpose models have shown promise, they often rely on a single or two combined architectures, which limits their ability to handle diverse degradation types effectively. RestorMixer addresses this by strategically fusing these architectures.

![RestorMixer](https://github.com/ClimBin/RestorMixer/blob/main/assets/pipeline.png "Pipeline")


-----

## ✨ Key Features

  - **Heterogeneous Architecture Fusion**: RestorMixer is the first model to effectively integrate CNN, Mamba, and Transformer architectures into a single, cohesive framework for image restoration.
  - **Multi-Stage Encoder-Decoder**: A hierarchical design processes features at different resolutions, from high-resolution local details to low-resolution global contexts.
  - **Superior Performance**: Achieves leading performance on multiple IR tasks, including deraining, desnowing, and super-resolution, for both single and mixed degradations.
  - **High Inference Efficiency**: The model is designed for speed, ensuring high performance without sacrificing computational efficiency.
  - **Versatility**: This framework is also applicable to other vision tasks (using an encoder-decoder structure).
-----

## 🚀 Getting Started

To get started with RestorMixer, follow the instructions below.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ClimBin/RestorMixer.git
    cd RestorMixer
    ```

-----

## 📊 Results
Click to expand and view the performance of RestorMixer across various image restoration tasks.

<details>
<summary>View Example Results</summary>
<p>
Here are some visual comparisons demonstrating RestorMixer's effectiveness:
<br>
</p>
<div align="center">
  <img src="https://github.com/ClimBin/RestorMixer/blob/main/assets/rain_res.png" alt="Deraining" width="1000"/>
  
<!--   <p>Figure 1: Comparison on a deraining task.</p> -->
</div>
<div align="center">
  <img src="https://github.com/ClimBin/RestorMixer/blob/main/assets/rain_vis.png" alt="Deraining_vis" width="1000"/>
  
<!--   <p>Figure 1: Comparison on a deraining task.</p> -->
</div>
<div align="center">
  <img src="https://github.com/ClimBin/RestorMixer/blob/main/assets/gopro_res.png" alt="Deblurring" width="500"/>
  
<!--   <p>Figure 1: Comparison on a deraining task.</p> -->
</div>
<div align="center">
  <img src="https://github.com/ClimBin/RestorMixer/blob/main/assets/gopro_vis.png" alt="Deblurring_vis" width="1000"/>
  
<!--   <p>Figure 1: Comparison on a deraining task.</p> -->
</div>

<div align="center">
  <img src="https://github.com/ClimBin/RestorMixer/blob/main/assets/combined_vis.png" alt="Combined_vis" width="1000"/>
  
<!--   <p>Figure 1: Comparison on a deraining task.</p> -->
</div>

  
</details>


-----

## 📄 Paper & Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gu2025efficient,
  title={An Efficient and Mixed Heterogeneous Model for Image Restoration},
  author={Gu, Yubin and Meng, Yuan and Zheng, Kaihang and Sun, Xiaoshuai and Ji, Jiayi and Ruan, Weijian and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2504.10967},
  year={2025}
}
```

-----

## 🙏 Acknowledgements

We would like to thank all the researchers whose work has inspired this project and the open-source community for providing invaluable tools and resources. 
Thanks to [VMamba](https://github.com/MzeroMiko/VMamba), [Restormer](https://github.com/swz30/Restormer), and all other works inspiring us.

-----
