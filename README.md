# Tree-based NN For Regression Analysis

This project aims to analyze errors in neural network predictions using tree-based models. It is based on the paper **"Analyzing the Behaviour of Tree-Based Neural Networks in Regression Tasks"**, currently in the submission process. The goal is to understand and improve the performance of neural networks by leveraging tree-based insights.


## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data and place it in the `data` directory.
2. Run the scripts in the `scripts` directory to train models and analyze errors.

Example command:
```bash
python ./src/train.py \
        --train_data_dir ${DATA_DIR} \
        --seed ${SEED} \
        --epochs 10 \
        --test_size 0.2 \
        --lr 1e-4 \
        --batch_size 4 \
        --train_on ${TRAIN_MODE} \
        --output_dir "${OUTPUT_DIR}" \
        --max_seq_length 2048 \
        --d_model 768 \
        --n_head 8 \
        --d_ff 2048 \
        --n_layer 1 \
        --drop 0.1 \
        --do_log
```


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{Tree_based_NN_Regression_analysis,
  author = {},
  title = {Dual-Transformer Architecture for Cross-Modal Learning on Tree-Structured Data in Regression Tasks},
  year = {},
  journal = {},
  note = {Under submission},
}
```
