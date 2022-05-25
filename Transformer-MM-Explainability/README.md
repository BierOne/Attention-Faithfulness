## Reproduction of Results

1. Pls follow the original [repo](https://github.com/hila-chefer/Transformer-MM-Explainability) to prepare required data and environments. Note that if you use lxmert, you can extract features by yourself following our [extract_imgfeature.sh](https://github.com/BierOne/Attention-Faithfulness/blob/master/Transformer-MM-Explainability/scripts/lxmert/gqa/extract_imgfeature.sh).

2. Run the shell script in [scripts folder](https://github.com/BierOne/Attention-Faithfulness/tree/master/Transformer-MM-Explainability/scripts) to generate explanations. For example,
    ```
    bash ./scripts/visualBert/vqa/save_exp.sh
    ```

3. Run the shell script in [scripts folder](https://github.com/BierOne/Attention-Faithfulness/tree/master/Transformer-MM-Explainability/scripts) to evaluate the faithfulness (violation, auctp, etc.) of explanations. For example,
    ```
    bash ./scripts/visualBert/vqa/run_perturbation.sh
    ```

4. \[Optional\] Run the notebook (evaluate.ipynb) to check results.



Credits

This codebase is based on this [repo](https://github.com/hila-chefer/Transformer-MM-Explainability).

* VisualBERT implementation is based on the [MMF](https://github.com/facebookresearch/mmf) framework.
* LXMERT implementation is based on the offical [LXMERT](https://github.com/airsplay/lxmert) implementation and on [Hugging Face Transformers](https://github.com/huggingface/transformers).
