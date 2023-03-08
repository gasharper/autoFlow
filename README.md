# autoFlow
**A sequence normalization flow framework with memory saving, automatic Jacobian tracking, and object-oriented programming features.**

- **Memory saving.** It has memory saving properties within and between blocks, just as simple as the [memcnn](https://github.com/silvandeleemput/memcnn) library.
- **Automatic Jacobian tracking.** Based on custom modules, automatic computation of log Jacobian determinant is implemented, just as simple as the [FrEIA](https://github.com/vislearn/FrEIA) library.
- **Object-oriented programming.** Using Python's object-oriented programming features, it's easy to construct reversible neural networks based on custom components.

To best of our knowledge, this is the first normalization flow (reversible neural network) framework that implements memory saving and automatic Jacobian tracking. The entire framework consists of only one file, *autoFlow.py*, which is easy to use and requires no installation!

Coming Soon……


**Note:** the *autoFlow* framework is the core framework used in our work (PyramidFlow, CVPR 2023), which is more powerful and user-friendly than memcnn or FrEIA. If it is helpful, please cite our work.
```
@misc{lei2023pyramidflow,
      title={PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow}, 
      author={Jiarui Lei and Xiaobo Hu and Yue Wang and Dong Liu},
      year={2023},
      eprint={2303.02595},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



