Implement PPFL in the practical setting. (based on [PFL](https://github.com/facebookresearch/FL_partial_personalization))

## Usage

### Data Generation

For generating data, see the `README.md` file in `dataset_statistics` folder.

We implement our methods with two federated practical dataset: EMNIST and Stackoverflow.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| EMNIST   |     Image classification       |     Resnet  |
| Stackoverflow   |     Next word prediction        |      Transformer |

### Pipeline
We carry out the same Pipeline as [PFL](https://github.com/facebookresearch/FL_partial_personalization)
- Pretrain in a non-personalized manner with FedAvg (PPFL for our method), see `./script/pretrain`;
- Training for personalization for FedAlt and pFedMe, see `./script/train`;
- Local Finetuning, seed `./script/finetune`.


