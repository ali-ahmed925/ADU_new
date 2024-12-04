# Domain Forgetting

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data Preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

## Model Zoo

### Vision-Language prompting methods
| Name  (configs)                                                                       |                                                             Model checkpoints                                                             |
|---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------:|
| [Independent V-L Prompting](configs/trainers/IVLP/vit_b16_c2_ep20_batch4_4+4ctx.yaml) | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/EuIwh-yMh_JBqB2Y_o8Jl14BPDKDRHC0JBPE1BugIeZiSQ?e=AJ8MhY) |
| [PromptSRC](configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml)            | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/EqFXPs2Zl9pKp39w3SqlR7QBDACTv-AgCXH6_cGflrUFwg?e=l33EBA) |


## Evaluation 
Please refer to the [EVAL.md](docs/EVAL.md) for detailed instructions on using the evaluation scripts and reproducing the official results using our pre-trained models.

## Training 
Please refer to the [TRAIN.md](docs/TRAIN.md) for detailed instructions on training PromptSRC and IVLP baseline from scratch.


<hr />

## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@InProceedings{Khattak_2023_ICCV,
    author    = {Khattak, Muhammad Uzair and Wasim, Syed Talal and Naseer, Muzammal and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
    title     = {Self-regulating Prompts: Foundational Model Adaptation without Forgetting},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {15190-15200}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at uzair.khattak@mbzuai.ac.ae or syed.wasim@mbzuai.ac.ae.


## Acknowledgements

Our code is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), along with [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

