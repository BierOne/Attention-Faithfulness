## Reproduction of Results

1. Pls follow the original [repo](https://github.com/akashkm99/Interpretable-Attention) to prepare required data and environments. Note that all datasets should be firstly preprocessed by runing .ipynb files in *preprocess folder*.

2. Run the shell script in [scripts folder](https://github.com/BierOne/Attention-Faithfulness/tree/master/Transparency/scripts) to **train the model**. For example,
    ```
    bash ./scripts/bc_all.sh
    ```

3. Run the notebook (*nlp_baseline_check_bc.ipynb* or *nlp_baseline_check_qa.ipynb*) to evaluate explanations and check results.
