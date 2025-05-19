# satellite-pose-estimation
Semantic Keypoint Detection and Pose Estimation for Satellites
This is the official implementation of papers

[Revisiting Monocular Satellite Pose Estimation With Transformer](https://ieeexplore.ieee.org/document/9743649)

[Monocular Satellite Pose Estimation Based on Uncertainty Estimation and Self-Assessment](https://ieeexplore.ieee.org/document/10633854)

## Overall architecture
![Overall architecture of proposed monocular pose estimation approach. Given an input image, the satellite is first detected and then the image
is cropped and resized. A random zoom-in technique is applied during training to reduce the impact of detection instability on the keypoint-set
predictor. The transformer-based keypoint-set predictor takes a fixed-size image as input and outputs the set of predefined keypoints. The pose of the
satellite is then estimated using PnP with the predefined 3-D points.](assert/revist_overall.png)

![Overall architecture of the proposed monocular pose estimation approach. Given an input image, the satellite is first cropped and resized. The
transformer-based keypoint-set predictor takes a fixed-size image as input and outputs the set of predefined keypoints and their Gaussian distributions.
The pose of the satellite is then estimated using PnP based on the 2-D–3-D correspondences. The self-assessment mechanism filters out unreliable
pose estimation results.]()

## Citation
If you use our papers in your work, please use the following BibTeX entries:

```
@ARTICLE{9743649,
  author={Wang, Zi and Zhang, Zhuo and Sun, Xiaoliang and Li, Zhang and Yu, Qifeng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Revisiting Monocular Satellite Pose Estimation With Transformer}, 
  year={2022},
  volume={58},
  number={5},
  pages={4279-4294},
  keywords={Satellites;Transformers;Pose estimation;Computational modeling;Training;Three-dimensional displays;Task analysis;Deep learning;keypoints;pose estimation;satellites;transformer},
  doi={10.1109/TAES.2022.3161605}}

@ARTICLE{10633854,
  author={Wang, Jinghao and Li, Yang and Li, Zhang and Wang, Zi and Yu, Qifeng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Monocular Satellite Pose Estimation Based on Uncertainty Estimation and Self-Assessment}, 
  year={2024},
  volume={60},
  number={6},
  pages={9163-9178},
  keywords={Uncertainty;Pose estimation;Satellites;Accuracy;Space vehicles;Semantics;Heating systems;Deep learning;keypoint detection;pose estimation;satellites;self-assessment;transformer},
  doi={10.1109/TAES.2024.3441569}}
```
