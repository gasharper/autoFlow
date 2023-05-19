# autoFlow
**A sequence normalizing flow framework with memory saving, automatic Jacobian tracking, and object-oriented programming features.**

- **Memory saving.** It has memory saving properties within and between blocks, just as simple as the [memcnn](https://github.com/silvandeleemput/memcnn) library.
- **Automatic Jacobian tracking.** Based on custom modules, automatic computation of log Jacobian determinant is implemented, just as simple as the [FrEIA](https://github.com/vislearn/FrEIA) library.
- **Object-oriented programming.** Using Python's object-oriented programming features, it's easy to construct reversible neural networks based on custom components.

To best of our knowledge, this is the first normalizing flow (reversible neural network) framework that implements memory saving and automatic Jacobian tracking. The entire framework consists of only one file, *autoFlow.py*, which is easy to use and requires no installation!

## Requirements
PyTorch >= 1.9.0

## Quick Start
git clone this repository by 
```
git clone https://github.com/gasharper/autoFlow.git
```
In our repository, `autoFlow.py` is the core framework file, while the `simple_test.py` is a quick start python script to learn how to use this framework. 

You can run the following command to test and learn this framework:
```
python simple_test.py
```

In `simple_test.py` script, we built the simplest `PyramidFlow` (w/o Volume Normalization and other tricks, only two layer) as a test model. You can flexibly build your own model in a similar way. 

**Official implementation**: The Official implementation of PyramidFlow is released at [here](https://github.com/gasharper/PyramidFlow). If you have any issues in reproducing our work, please create a [new issue](https://github.com/gasharper/PyramidFlow/issues/new).

## Note
The *autoFlow* framework is the core framework used in our work (PyramidFlow, CVPR 2023), which is more powerful and user-friendly than memcnn or FrEIA. If it is helpful, please star this repository and cite our work.
```
@article{lei2023pyramidflow,
  title={PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow},
  author={Jiarui Lei and Xiaobo Hu and Yue Wang and Dong Liu},
  journal={CVPR},
  year={2023}
}
```



