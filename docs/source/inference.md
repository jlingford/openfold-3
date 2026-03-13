# OpenFold3 Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold3. OpenFold3 implements the features described in [AlphaFold3 *Nature* paper](https://www.nature.com/articles/s41586-024-07487-w).

This guide covers how to use OpenFold3 to make structure predictions.


## 1. Inference features

OpenFold3 replicates the full set of input features described in the *AlphaFold3* publication. All features of AlphaFold3 are **fully implemented and supported in training**. We are actively working on integrating the same functionalities into the inference pipeline. 

Below is the current status of inference feature support by molecule type:


### 1.1 Protein

Supported:

- Prediction with MSA
    - using ColabFold MSA pipeline
    - using pre-computed MSAs
- Prediction without MSA
- OpenFold3's own MSA generation pipeline
- Template-based prediction
    - using ColabFold template alignments
    - using pre-computed template alignments
- Non-canonical residues

Coming soon:

- Covalently modified residues and other cross-chain covalent bonds
- User-specified template structures (as opposed to top 4)

### 1.2 DNA

Supported:

- Prediction without MSA (per AF3 default)
- Non-canonical residues

Coming soon:

- Covalently modified residues and other cross-chain covalent bonds


### 1.3 RNA

Supported:

- Prediction with MSA, using OpenFold3's own MSA generation pipeline
- Prediction without MSA
- OpenFold3's own MSA generation pipeline
- Non-canonical residues

Coming soon:

- Template-based prediction
- Covalently modified residues and other cross-chain covalent bonds
- Protein-RNA MSA pairing


### 1.4 Ligand

Supported:

- Non-covalent ligands

Coming soon:

- Covalently bound ligands
- Polymeric ligands such as glycans


## 2. Pre-requisites

- OpenFold3 Conda Environment. See [OpenFold3 Installation](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/Installation.md) for instructions on how to build this environment.
- OpenFold3 Model Parameters. See {ref}`OpenFold3 Setup <setup-openfold3-parameters>` for an easy option to download model parameters.