# A-Blockchain-Based-Decentralized-Federated-Learning-Framework-
with Adaptive Client Trust for Privacy-Preserving IoT Security
# Hybrid Blockchain--Federated Learning Framework for Privacy-Preserving IoT Security

This repository contains the Google Colab/Jupyter implementation used
for the research work:

**A Hybrid Blockchain--Federated Learning Framework with AI-Based
Anomaly Detection for Privacy-Preserving IoT Security**

The implementation uses the **CIC IoT-DIAD 2024** dataset and develops a
hybrid IoT intrusion-detection framework combining centralized AI
benchmarking, non-IID federated learning, class-aware performance
refinement, blockchain-assisted trust management, security stress
testing, statistical validation, and reproducibility packaging.

## Dataset

The experiments are based on the **CIC IoT-DIAD 2024** dataset.

The dataset itself is not redistributed in this repository. Download it
from the official dataset provider and place it in Google Drive before
running the notebook. The dataset path used in the notebook can be
adjusted to match the user's Drive structure.

## Recommended Environment

The notebook is designed primarily for **Google Colab** with Google
Drive mounted for persistent storage. A GPU runtime is recommended for
the deep-learning and federated-learning experiments.

The code automatically installs or checks several required packages
where needed. Major libraries used across the workflow include:

-   Python
-   NumPy
-   Pandas
-   PyTorch
-   scikit-learn
-   XGBoost
-   LightGBM
-   imbalanced-learn
-   Matplotlib
-   SciPy
-   PyArrow

Additional packages are installed by individual notebook stages when
required.

## Experimental Workflow

The notebook is organized as a sequential experimental pipeline. Saved
outputs from earlier stages are reused by later stages so that completed
experiments do not need to be repeated unnecessarily.

### Step 1 --- Complete Dataset Preparation Pipeline

Step 1 prepares the CIC IoT-DIAD 2024 dataset and experimental
environment.

It performs Google Drive and GPU setup, connects to the previously
downloaded dataset, creates a new project directory structure, audits
the dataset files and schemas, prepares clean data, generates
model-ready data partitions, and saves the artifacts required by
subsequent stages.

The main project directory used by the notebook is:

`/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026/`

### Step 2 --- Centralized AI Baseline Benchmarking

Step 2 establishes centralized machine-learning/deep-learning baselines.

The evaluated models include:

-   Random Forest
-   XGBoost
-   Multilayer Perceptron (MLP)
-   1D Convolutional Neural Network (1D-CNN)

The experiments cover binary anomaly detection and multiclass intrusion
classification.

### Step 2B --- Class-Imbalance Mitigation and Robust Centralized AI

This stage introduces literature-guided class-imbalance handling while
preserving the original Step 2 baseline results.

Important experimental controls include:

-   feature ranking using training data only;
-   resampling/SMOTE on training data only;
-   model, feature-count, and threshold selection using validation data;
-   preservation of the held-out test set for final evaluation.

### Step 2C --- Scientific Split Repair and FL-Ready Baseline

Step 2C preserves the earlier source-IP-disjoint experiment as an
out-of-distribution stress test while creating a separate benchmark
suitable for centralized-versus-federated comparison.

This stage addresses the distinction between source-IP metadata and
verified physical-device identity and creates an FL-ready evaluation
protocol.

### Step 3 --- 10-Client Non-IID Federated Learning

Step 3 establishes the federated-learning baseline using **10 non-IID
clients**.

The principal FL algorithms include:

-   FedAvg
-   FedProx

A matched centralized control is retained for comparison.

### Step 3A --- SOTA-Oriented Non-IID FL Refinement

Step 3A refines the non-IID federated IDS while retaining the completed
Step 3 results.

The stage introduces a more advanced class-aware federated-learning
route intended to improve minority-class performance without using
validation or test information for training.

### Step 3B --- FedAdam, FedLC and Feature-Token Transformer

This stage evaluates an additional federated optimization and
representation-learning configuration combining:

-   FedAdam
-   FedLC
-   Feature-Token Transformer

The validation and test partitions remain protected from training.

### Step 3C-V3 --- Literature-Comparable and IID-FL Benchmark

Step 3C-V3 constructs an additional literature-comparable protocol and a
high-accuracy IID federated-learning benchmark.

This experiment is retained separately from the stricter non-IID
protocol so that results obtained under different evaluation assumptions
are not treated as directly equivalent.

## Step 4 --- Proposed Model Development

The Step 4 family contains the performance-development stages leading to
the final proposed hybrid framework.

### Step 4A --- PCH-FL

Step 4A introduces the **Prototype-Calibrated Hierarchical Federated IDS
(PCH-FL)** as a performance backbone.

### Step 4.4A and R2--R6 --- Performance Refinement

A sequence of controlled refinement experiments evaluates
literature-informed and class-aware mechanisms for improving federated
multiclass intrusion detection.

These stages include:

-   LT-SiamTAC-FL
-   AHF-RCE
-   CARF-Stack
-   FEC-Router
-   FCS-MoE
-   HFSB-FL

The refinement sequence is used as the architecture/performance
development phase before freezing the final framework.

### Step 4B --- BC-ATG-HFSB-FL

Step 4B integrates the performance backbone with blockchain-assisted
adaptive trust gating to form:

**BC-ATG-HFSB-FL --- Blockchain-Assisted Adaptive Trust-Gated
Hierarchical Federated Learning**

At this point, the architecture-search/performance-only refinement is
stopped.

### Step 4C --- Final Proposed BC-ACTG-HFSB-FL Model

Step 4C defines the final proposed framework:

**BC-ACTG-HFSB-FL --- Blockchain-Assisted Adaptive Contribution
Trust-Gated Hierarchical Federated Learning**

This stage refines the trust mechanism using contribution-aware
information and produces the frozen model used for final validation.

## Step 5 --- Final Validation, Ablation and Statistical Analysis

After the proposed architecture is frozen, Step 5 performs final
validation rather than another architecture search.

The stage evaluates the complete framework and its components through
ablation and statistical analysis. It is intended to quantify the
contribution of the proposed mechanisms and assess the stability of the
final model.

## Step 6 --- Security Stress Test and Threat-Model Validation

Step 6 evaluates the frozen **BC-ACTG-HFSB-FL** framework under
adversarial/security-oriented conditions.

The purpose is to assess whether the trust-gating and
blockchain-assisted security mechanisms remain effective under the
defined threat model rather than only under clean classification
conditions.

## Step 7 --- Paper-Ready Result Consolidation and Reproducibility Package

Step 7 does **not** retrain, tune, or modify the final model.

It consolidates the saved experimental evidence from the completed
model-development, validation, and security stages into paper-ready
outputs and a reproducibility package.

## Step 8 --- Publication Figures and Statistical Validation

Step 8 is a post-training result-analysis stage.

It reads the saved results from the final experiments and generates
publication-oriented figures, statistical validation outputs, and
archived result packages. It does not retrain or retune the proposed
model.

A final safe-download packaging block creates verified ZIP archives of
the Step 8 outputs.

## Final Proposed Framework

The final model used for validation is:

**BC-ACTG-HFSB-FL**

**Blockchain-Assisted Adaptive Contribution Trust-Gated Hierarchical
Federated Learning**

At a high level, the framework combines:

**IoT Traffic → Data Preparation → AI-Based Intrusion Detection →
Federated Learning → Hierarchical/Class-Aware Learning → Adaptive
Contribution Trust Gating → Blockchain-Assisted Audit/Security → Final
Validation**

## Experimental Design Principles

The notebook applies several controls intended to support reproducible
and defensible evaluation:

-   training-only preprocessing and resampling where specified;
-   validation-based model and threshold selection;
-   held-out test evaluation;
-   separation of strict non-IID/OOD protocols from
    literature-comparable/IID benchmarks;
-   preservation of earlier experimental outputs rather than silently
    overwriting them;
-   freezing of the final proposed architecture before final validation
    and security stress testing;
-   saved intermediate artifacts in Google Drive;
-   result consolidation without retraining in the final reporting
    stages.

## How to Run

1.  Open the `.ipynb` notebook in Google Colab.
2.  Select a GPU runtime when available.
3.  Mount Google Drive.
4.  Download the CIC IoT-DIAD 2024 dataset separately.
5.  Update the dataset path in Step 1 if your Google Drive location
    differs from the notebook configuration.
6.  Run the notebook sequentially from Step 1 onward.
7.  Allow each stage to save its outputs before running a dependent
    later stage.
8.  Do not use the final reporting/figure-generation stages as
    model-development stages.

Because several later cells reuse artifacts generated by earlier cells,
running the notebook in sequence is recommended for a clean
reproduction.

## Repository Files

A recommended repository structure is:

``` text
.
├── README.md
├── kamran_paper_3.ipynb
├── requirements.txt
├── LICENSE
└── results/
    └── optional exported paper-ready results
```

For public release, the notebook may be renamed to a descriptive
filename such as:

`BC_ACTG_HFSB_FL_CIC_IoT_DIAD_2024.ipynb`

## Reproducibility Notes

The implementation stores intermediate and final artifacts under the
project directory in Google Drive. This makes it possible to resume
later stages without repeating all previous computation.

For the strongest reproducibility:

-   use the same dataset release;
-   retain the random seeds specified in the notebook;
-   use the same train/validation/test protocols;
-   avoid changing thresholds after observing test results;
-   report which experimental protocol produced each table or figure;
-   keep the final model frozen during Steps 5--8.

## Data Availability

The CIC IoT-DIAD 2024 dataset is obtained separately from its official
provider and is not included in this repository.

## Code Availability

The notebook in this repository contains the experimental implementation
used to generate the reported results. Intermediate datasets, model
checkpoints, and large generated artifacts may be excluded from GitHub
because of storage limitations.

## Citation

If you use this implementation, please cite the associated research
paper after publication.

``` text
Citation information will be updated after publication.
```

## License

See the `LICENSE` file in this repository for the applicable software
license.

## Disclaimer

This code is provided for academic research and reproducibility. Results
may vary with hardware, software-library versions, dataset placement,
and runtime configuration.
