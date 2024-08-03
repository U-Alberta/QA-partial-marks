Official implementation of "[Accurate and Nuanced Open-QA Evaluation Through Textual Entailment](https://arxiv.org/abs/2405.16702)"

[**[Paper]**](https://arxiv.org/abs/2405.16702) [**[Video]**](https://vimeo.com/994611659) [**[Poster]**](./poster.pdf)

If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{2024yao_qaeval,
    title = "Accurate and Nuanced Open-{QA} Evaluation Through Textual Entailment",
    author = "Peiran Yao  and
      Barbosa, Denilson",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    url = "https://arxiv.org/abs/2405.16702",
}
```

## Setting up
A valid OpenAI API key is required.
```bash
export OPENAI_API_KEY=your_openai_api_key
pip install openai backoff pandas tqdm
# If you are interested in finetuning
pip install torch transformers datasets peft trl
```

## Running QA evaluation
Two steps are needed to evaluate the correctness of question-answer pairs:

```bash
# Convert QA pairs to statements
python3 qa2s-gpt.py --dataset [NQ|TQ]
# Run entailment test between system answer and reference answer
python3 nli-gpt.py --dataset [NQ|TQ]
```

The notebook `eval-eval.ipynb` is for evaluating the QA evaluation results.

Scripts for training baselines involving finetuning are in the `trained-eval` and `rev` folders.

## Getting partial and bonus marks
Running `python3 cot.py` to use GPT-3.5 to verbalize the entailment reasoning. The notebook `cot_vs_pr.ipynb` computes a score from the verbalized reasoning and evaluates the score by AUROC.