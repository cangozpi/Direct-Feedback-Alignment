# Direct Feedback Alignment

This code implements Direct Feedback Alignment (DFA) functionality on top of pytorch. The aim of this repository is to investigate the affects of different normalization and activation functions on DNN training while using DFA. Current implementation does not support parallelization capabilities of DFA, and it only works for Linear layers at the moment.

---

## Configure & Run Code:

**Important:** Check out _config.yaml_ file for configuring the hyperparameters and the regularizations used.

- After setting parameters in _config.yaml_ then run the code by:

  ```bash
  python main.py
  ```

- To see visualizations of your experiments (_TensorBoard_):
  ```bash
  tensorboard --logdir logs
  ```
  **Note:** _Your experiments will be automatically logged under the /logs directory. Your config.yaml configurations will be available under the "HPARAMS" tab in TensorBoard._

---

## Supported Regularizations:

1. Dropout
2. BatchNorm
3. LayerNorm
4. Weight Decay
5. L2 Regularization
6. L1 Regularization
7. Learning Rate Scheduling (ConstLR)
8. Early Stopping (_Implemented yet not supported by training on MNIST dataset_)

---

## TODO:

- Setup dataset :heavy_check_mark:
- Implement a Neural Network :heavy_check_mark:
- DFA backward implementation :heavy_check_mark:
- Implement train & test loops :heavy_check_mark:
- Implement Normalization functions :heavy_check_mark:
- Add config file for experiments :heavy_check_mark:
- Add tensorboard visualizations :heavy_check_mark:
- Perform experiments with Regularizations
- Add experiments for different activation functions
- Implement parallelization capability for backward pass
- Add another Dataset and Model
- Perform experiments with Activations
