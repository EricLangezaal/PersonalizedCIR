# PersonalizedCIR
This is a reproduction study of the original paper: ["How to Leverage Personal Textual Knowledge for Personalized Conversational Information Retrieval."](https://arxiv.org/abs/2407.16192). The original codebase can be found [here](https://github.com/fengranMark/PersonalizedCIR). This codebase has been substantially extended to allow for self-contained experiments.

# Environment and dependencies
We highly suggest using a Conda environment through the provided installation script, since multiple requirements do not install correctly from PyPI automatically. To run this script, please invoke the following in the repositories main folder. This script will automatically install Anaconda if it cannot load nor find Conda.
```bash
bash install_environment.sh
```

If you want to install the dependencies manually, please see the list below.
Main packages:
- python 3.11.x
- torch 2.5.1
- transformers 4.46.2
- numpy 2.1.3
- faiss-gpu 1.9.0
- pyserini 0.42.0
- openai 1.54.3
- pytrec-eval 0.5
- toml 0.10.2
- tenacity 9.0.0
- accelerate 0.26.2


# Preparation

## 1. Downloading data

This repository already contains the 2023 TREC iKAT conversation data inside the [data](/data/) folder. The 116M document collection has to be downloaded from the [iKAT TREC website](https://ikattrecweb.grill.science/UvA/). Note that the document collection is not public domain, requiring an account to be accessed. This repository contains Slurm job scripts to download both the raw JSONL passage data, as well as the prepared BM25 index. If you are not using Snellius Slurm, please also change the output directory on top of the job script. Running these scripts directly from terminal is possible by coping their contents to a bash script.

> [!IMPORTANT]
>  Both scripts below require an account to download the collection files. Please make sure to set the `IKAT_USERNAME` and `IKAT_PASSWORD` variables prior to downloading, for example by defining these in a `set_secrets.sh` script.  

The passages can be downloaded into a single `collection.jsonl` file using the script below. These files are only needed for the ANCE/dense retrieval. Make sure to modify the destination directory as needed.
```bash
sbatch jobs/download/download_raw_dataset.job
```

To download the BM25 index for Pyserini use the script below, this is only required for BM25. Make sure to modify the destination directory as needed.
```bash
sbatch jobs/download/download_index_dataset.job
```

## 2. Preprocessing
After downloading the data, the JSONL collection needs to be preprocessed for the ANCE dense retrieval tasks. 
The pre-trained ad-hoc search model ANCE is used generate passage embeddings, and is hosted by us on [HuggingFace](https://huggingface.co/3ricL/ad-hoc-ance-msmarco). The entire preprocessing can be invoked using the following two Python scripts. Note that this can take multiple days to run. Make sure to modify the filepaths in both configuration files to correspond to your file system.
```bash
python index/gen_tokenized_doc.py --config=index/gen_tokenized_doc.toml
python index/gen_doc_embeddings.py --config=index/gen_doc_embeddings.toml
```

**Optional**: After the preprocessing has finished, it is possible to validate if this has been done correctly:
```bash
python index/verify_outputs.py --config=index/gen_tokenized_doc.toml
```

**Optional**: Later reproduction stages required flattened versions of the iKAT conversation data. These are already present in this repository, specifically [2023_test_topics_flattened.jsonl](data/2023_test_topics_flattened.jsonl). Should you want to recreate this file, this can be done through:
```bash
python pcir/preprocessing_data.py
```

## 3. iKAT TREC 2024
We have also repeated our research for the new iteration of the iKAT TREC dataset, and we host the conversation data for this year's iteration in the [data](/data/) folder too. Note that the 2024 gold standard relevance file (2024-qrels.all_turns.txt) is not yet public, so it has not been included in this repository. So while our entire pipeline supports this dataset too, you have to obtain this file yourself once it has been published, and put it in the data-directory. For clarity, the default arguments and examples below will assume the 2023 dataset, so the command line/configuration arguments need to be changed for 2024.

# Reproduction
This section will outline how to reproduce every experiment, provided the datasets have been downloaded and preprocessed. Firstly the basic steps will be outlined, after which we will briefly elaborate how to run ablations such as in-context learning, using different LLM's or using OpenAI batch processing for efficiently repeating experiments. 

## 0. Dataset statistics
It is possible to automatically calculate some rudimentary statistics of the dataset, which form the basis of the table in our paper. This can be done by running:
```bash
python data/generate_data_table.py
```

## 1. Query reformulation
> [!NOTE]
>  If you do not want to run the query reformulation process yourself, it is possible to use the files we created, which are hosted in [this repository](data/results/) too (including subfolders for the in-context learning or Llama). Note that these are output files of the reformulation already, so this entire section can be skipped when using those. 

> [!NOTE]
> The query reformulation process inherently introduces differences between runs, as OpenAI's API will give different results for the same query. To get our exact results, please use the files already hosted in this repository.

The method of this research distinguishes between two different pipelines: Firstly, there are approaches which separately select PTKB (either intelligently or through a baseline) and then reformulate using an LLM. These approaches can be reproduced through section 1.1. Next, there is the Select And Reformulate (SAR) pipeline, which does both PTKB selection and reformulation in one pass, as explained in section 1.2. The prompt templates are provided in [prompt_template.md](prompt_template.md). All scripts output files according to a standard naming scheme (in the [data](data/) folder), such that filepaths often don't have to be specified. If this doesn't work, each script also accepts an input and output file overwrite.

### 1.1 Two stage approaches.
Firstly, the PTKB can be selected. There are five approaches as detailed in the paper: All, None, Human, Automatic and LLM (STR). 
- **None** (no PTKB): ```python pcir/methods/reformulate.py --annotation 'None"```
- **All** (use all PTKB): ```python pcir/methods/reformulate.py --annotation 'All"```
- **Human**: ```python pcir/methods/reformulate.py --annotation 'Human"```
- **STR**: First run ```python pcir/methods/select_ptkb_Xshot.py --shot 0``` to select the relevant PTKB. Then run ```python pcir/methods/reformulate.py --annotation 'STR' --shot 0```
- **Automatic**:  ```python pcir/methods/ptkb_automatic_method.py```
      
### 1.2 Select and reformulate (SAR)
To run the SAR pipeline, which selects PTKB and reformulates the query in a single pass, the following can be used:
```bash
python pcir/methods/select_reformulate_Xshot.py --shot 0
```

## 2. Retrieval evaluation
Once the self-contained queries have been obtained through LLM reformulation, sparse or dense retrieval can be employed to evaluate the quality of the queries.

### 2.1 Sparse retrieval
We can perform sparse retrieval to evaluate the personalized reformulated queries by running for example. For a JSONL file obtained through the automatic method, make sure to use the `--automatic_method` flag.
```bash
python pcir/eval/bm25_ikat.py --input_query_path data/results/2023_test_SAR_0shot.jsonl --index_dir_path path/to/bm25/index
```
    
### 2.2 Dense retrieval
We can perform dense retrieval to evaluate the personalized reformulated queries by running:
```bash
python pcir/eval/ance_ikat.py --config pcir/eval/ance_ikat_config.toml
```
You will need to modify the 'passage_offset2pid_path' and 'passage_collection_path' in this configuration file accordingly. For a JSONL file obtained through the automatic method, make sure to use the `--automatic_method` flag too.

## 3. In context learning
The procedure to run in-context learning does not differ much from the pipelines outlined above. Just make sure to run either STR or SAR using the `--shot x` flag in all scripts from section 1 of this part, where 'x' denotes the number of in context learning examples (can be 0, 1, 3 or 5 currently) from the [2023 training dataset](data/2023_train_topics.json) (Note that a 2024 train set does not exist, so we used to 2023 examples for 2024 too). The retrieval evaluation is identical, using the reformulation jsonl file obtained with multiple shots. Note that the original paper always used the same fixed examples, but we also support using random examples through the ```--random_examples``` flag both SAR and STR.

## 4. Using a different LLM
The original paper used only OpenAI's `gpt-3.5-turbo-16k`. We extended this by also implementing `gpt-4o-mini` and `Llama-3.1 8B`. To use a different LLM, simply add the command line arguments `--llm_model model_string` for any script from section 1 of this part. If the model string is not a GPT version, it is assumed to be a HuggingFace identifier. This allows us to do `--llm_model "meta-llama/Meta-Llama-3.1-8B-Instruct"` to use `Llama-3.1 8B`. 

## 5. Repeating experiments with batch processing 
As an extension, we allow experiments to be repeated multiple times using OpenAI's cheaper batch processing API. All files related to this reformulation can be found in the [batch_processing](pcir/batched_processing/) folder. There are four `submit` scripts, which submit jobs for the SAR method, the LLM-based PTKB selection (STR), any the reformulation required for All, None, Human and STR. There is also a submission file for the automatic method. 

Next, there are four similar scripts to check if the corresponding job has completed and save it to a file if applicable. Use `batch_cancel.py` to cancel a job. 

### 5.1 Aggregation
If you have run an experiment multiple times, you can aggregate the results to calculate means and standard deviations. Use `--automatic_method` if it concerns an automatic method run. Note that this script assumes the files to be named `{input}_run{i}.jsonl`, where `i` goes from 1 to the number of runs specified (5 by default).
```bash
python pcir/eval/aggregate_results.py --input data/batch/gpt-4o-mini/processed/batch_SAR_5shot --year "2023" --num_runs 5
``` 
This script will output an aggregated file, by default in the `data/batch/gpt-4o-mini/aggregated/` folder.

### Significance testing
Given multiple aggregation summary files, each from a different method, it can be interesting to perform a T-Test to see if certain methods differ significantly. This can be done through a dedicated script, where `n` denotes how many experiments were conducted for each aggregation file.
```bash
python pcir/eval/significance_test.py --folder basefolder --files file1.jsonl file2.jsonl file3.jsonl -n 5
```
This script will automatically test for significance across any pair of aggregation files, for any defined subset and metric.

