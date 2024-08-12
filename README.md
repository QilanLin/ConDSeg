
<h2 align="center"> ✨ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement</h2>

## Overview🔍
<div>
    <img src="figures/framework.jpg" width="96%" height="96%">
</div>

**Figure 1. The framework of the proposed ConDSeg.**


**_Abstract -_** Medical image segmentation plays an important role in clinical decision-making, treatment planning, and disease tracking. However, it still faces two major challenges. On the one hand, there is often a ``soft boundary'' between foreground and background in medical images, with poor illumination and low contrast further reducing the distinguishability of foreground and background within the image. On the other hand, co-occurrence phenomena are widespread in medical images, and learning these features is misleading to the model's judgment. To address these challenges, we propose a multimodal-generalized framework called Contrast-Driven Medical Image Segmentation (ConDSeg). First, we develop a Consistency Reinforcement training strategy dedicated to improving the Encoder's robustness in various illumination and contrast scenarios, enabling the model to extract high-quality features even in adverse environments. Second, we introduce a Semantic Information Decoupling (SID) module, which decouples features from the encoder into foreground, background, and uncertain regions, gradually acquiring the ability to reduce uncertainty during training. The Contrast-Driven Feature Aggregation (CDFA) module then uses the foreground and background information to guide multi-level feature fusion and key feature enhancement, further distinguishing between foreground and background. We also propose a Size-Aware Decoder to accurately locate individuals of different sizes in the image, thus avoiding erroneous learning of co-occurrence features. Extensive experiments on five medical image datasets across three modalities demonstrate the state-of-the-art performance of our method, proving its advanced nature and applicability to a wide range of medical image segmentation tasks.

## Datasets📚
To verify the performance and general applicability of our ConDSeg in the field of medical image segmentation, we conducted experiments on five challenging public datasets: Kvasir-SEG, Kvasir-Sessile, GlaS, ISIC-2016, and ISIC-2017, covering subdivision tasks across three modalities. 

| Dataset      | Modality                  | Anatomic Region | Segmentation Target | Data Volume |
|--------------|---------------------------|-----------------|---------------------|-------------|
| Kvasir-SEG   | endoscope                 | colon           | polyp               | 1000        |
| Kvasir-Sessile | endoscope               | colon           | polyp               | 196         |
| GlaS         | whole-slide image (WSI)   | colorectum      | gland               | 165         |
| ISIC-2016    | dermoscope                | skin            | malignant skin lesion | 1279       |
| ISIC-2017    | dermoscope                | skin            | malignant skin lesion | 2750       |

For Kvasir-SEG, we followed the official recommendation, using a split of 880/120 for training and validation. Kvasir-Sessile, a challenging subset of Kvasir-SEG, adopted the widely used split of 156/20/20 for training, validation, and testing as in [TGANet](https://github.com/nikhilroxtomar/TGANet), [TGEDiff](https://www.sciencedirect.com/science/article/pii/S0957417424004147), etc. For GlaS, we used the official split of 85/80 for training and validation. For ISIC-2016, we utilized the official split of 900/379 for training and validation. For ISIC-2017, we also followed the official recommendation, using a split of 2000/150/600 for training, validation and testing.

## Experimental Results🏆

[//]: # (![img.png]&#40;figures/comp_1.png&#41;)

[//]: # (![img.png]&#40;img.png&#41;)

**Table 1. Quantitative comparison of ConDSeg with state-of-the-art methods on Kvasir-Sessile, Kvasir-SEG and GlaS datasets.**
<div>
    <img src="figures/comp1.jpg" width="96%" height="96%">
</div>

**Table 2. Quantitative comparison of ConDSeg with state-of-the-art methods on ISIC-2016 and ISIC-2017 datasets.**
<div>
    <img src="figures/comp2.jpg" width="40%" height="40%">
</div>

<br> </br>

<div>
    <img src="figures/vis.jpg" width="96%" height="96%">
</div>

**Figure 2. Visualization of results comparing with other methods.**


## Getting Started🚀
### Data Preparation
The dataset should be organised as follows,taking Kvasir-SEG as an example:
```text
Kvasir-SEG
├── images
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   ├── cju0qoxqj9q6s0835b43399p4.jpg
│   ├── cju0qx73cjw570799j4n5cjze.jpg
│   ├── ...
├── masks
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   ├── cju0qoxqj9q6s0835b43399p4.jpg
│   ├── cju0qx73cjw570799j4n5cjze.jpg
│   ├── ...
├── train.txt
├── val.txt
```

### Training
- To train the first stage of ConDSeg, run: `train_stage1.py`.
- To train the second stage of ConDSeg, add the weights of the first stage to the `train.py` script and run it.

### Evaluation
- To evaluate the model and generate the prediction results, run: `test.py`.

### Another Version Using Transformer Encoder
If you are interested in the version of ConDSeg that uses the Pyramid Vision Transformer as the Encoder, please see `./network_pvt`.


## Cite our work📝
```bibtex
Coming soon.
```

## License📜
The source code is free for research and education use only. Any comercial use should get formal permission first.


