# Direct Feedback Alignment

This code implements Direct Feedback Alignment (DFA) functionality on top of pytorch. The aim of this repository is to investigate the affects of different normalization and activation functions on DNN training while using DFA. Current implementation does not support parallelization capabilities of DFA, and it only works for Linear layers at the moment.

---

## TODO:

- Setup dataset :heavy_check_mark:
- Implement a Neural Network :heavy_check_mark:
- DFA backward implementation :heavy_check_mark:
- Implement train & test loops :heavy_check_mark:
- Add experiments for different normalization functions
- Add experiments for different activation functions
- Add tensorboard visualizaitons
- Implement parallelization capability for backward pass
- Add another Dataset and Model
- Perform experiments
