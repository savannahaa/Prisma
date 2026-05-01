# PRISMA

Public code release for **PRISMA** and the retained baselines.

For compatibility, some internal variable names and artifact suffixes still use `paperfaithful_mainline`.

## Installation

Recommended Python version: `3.12`

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Workflow Summary

1. **Install dependencies**  
   Used here: `PyTorch`, `NumPy`, `Transformers`, `Sentence-Transformers`, `FAISS`, `scikit-learn`.

2. **Prepare a dataset as a PRISMA workset** with `src/prepare_assets.py`  
   Used here:
   - datasets: `MS MARCO`, `Amazon ESCI (US)`, `FIQA-2018`, `Cohere MSMARCO v2.1`
   - embedding model: `intfloat/e5-large-v2`
   - libraries: `PyTorch`, `Transformers`, `Sentence-Transformers`, `NumPy`

3. **Build offline clustering and indexing assets**  
   Used here:
   - `spherical k-means` for clustering
   - `HNSW` and `FAISS` for dense ANN indexing
   - `NumPy` and `scikit-learn` for array processing and clustering utilities

4. **Run the PRISMA online pipeline** with `src/client/run_online_pipeline.py`  
   Used here:
   - `E5` query embeddings
   - `FAISS HNSW` dense retrieval
   - `RDP` privacy mechanism
   - separated client/server deployment

5. **Run baselines if needed**  
   Used here:
   - `plaintext_ann` as the repo-local **Non-Private PRISMA** control baseline
   - `RemoteRAG`
   - `Panther`
   - `Tiptoe`
   - `TenSEAL` only for the retained encrypted baseline code paths (for example the `Tiptoe` CKKS path), not for `plaintext_ann`

## Deployment

Main entrypoint:

```bash
python src/prepare_assets.py --dataset <mse5|cohere|amzn|fiqa> ...
python src/client/run_online_pipeline.py
```

Notes:

- client and server roles are separated
- different GPUs can be used for client side and server side
- the upload-ready snapshot defaults to `32` CPU threads

`Panther` additionally needs the public OpenPanther repository.
Default path:

```text
external/OpenPanther
```

Optional override:

```bash
export OPENPANTHER_ROOT=/absolute/path/to/OpenPanther
```

## Implementation and Experimental Environment

- **PyTorch**  
  Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.  
  <https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library>

- **FAISS**  
  Douze et al., *The Faiss library*, arXiv 2024.  
  <https://huggingface.co/papers/2401.08281>

- **NumPy**  
  Harris et al., *Array programming with NumPy*, Nature 2020.  
  <https://www.nature.com/articles/s41586-020-2649-2>

- **Transformers**  
  Wolf et al., *Transformers: State-of-the-Art Natural Language Processing*, EMNLP Demos 2020.  
  <https://aclanthology.org/2020.emnlp-demos.6/>

- **Sentence-Transformers**  
  Reimers and Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019.  
  <https://huggingface.co/papers/1908.10084>

- **scikit-learn**  
  Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.  
  <https://www.jmlr.org/papers/v12/pedregosa11a.html>

- **E5 / `intfloat/e5-large-v2`**  
  Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training*, arXiv 2022.  
  <https://www.microsoft.com/en-us/research/publication/text-embeddings-by-weakly-supervised-contrastive-pre-training/>

## Indexing, Clustering, and Privacy Components

- **spherical k-means**  
  <https://jmlr.csail.mit.edu/papers/v6/banerjee05a.html>

- **HNSW**  
  Malkov and Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, TPAMI 2020.  
  <https://pubmed.ncbi.nlm.nih.gov/30602420/>

- **RDP**  
  Mironov, *Rényi Differential Privacy*, CSF 2017.  
  <https://research.google/pubs/r%C3%A9nyi-differential-privacy/>

## Datasets

- **MS MARCO**  
  Bajaj et al., *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*, 2016.  
  <https://huggingface.co/papers/1611.09268>  
  <https://www.microsoft.com/en-us/research/?p=328361>

- **Amazon ESCI (US)**  
  Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*, 2022.  
  <https://www.amazon.science/code-and-datasets/shopping-queries-dataset-a-large-scale-esci-benchmark-for-improving-product-search>  
  <https://arxiv.gg/abs/2206.06588>

- **BEIR**  
  Thakur et al., *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, 2021.  
  <https://huggingface.co/papers/2104.08663>

- **FIQA-2018**  
  Maia et al., *WWW’18 Open Challenge: Financial Opinion Mining and Question Answering*, WWW Companion 2018.  
  <https://huggingface.co/datasets/explodinggradients/fiqa>

- **Cohere MSMARCO v2.1**  
  <https://microsoft.github.io/msmarco/Datasets.html>  
  <https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3>

## Comparison Methods

- **plaintext_ann**  
  Repo-local **Non-Private PRISMA** control baseline.  
  This is the clean same-pipeline baseline used for comparison against PRISMA, without encrypted retrieval and without the privacy mechanism.

- **RemoteRAG**  
  *RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service*, Findings of ACL 2025.  
  <https://aclanthology.org/2025.findings-acl.197/>

- **Panther**  
  *Panther: Private Approximate Nearest Neighbor Search in the Single Server Setting*, CCS 2025.  
  <https://www.researchgate.net/publication/397882024_Panther_Private_Approximate_Nearest_Neighbor_Search_in_the_Single_Server_Setting>

- **Tiptoe**  
  <https://eprint.iacr.org/2023/1438>
