# GSAP-NER Data and code

### [link to project website](https://data.gesis.org/gsap/gsap-ner)

Find here:
 * dataset (incl. predictions of our compared models)
 * training scripts and configs
 * inference scripts and configs
 * evaluation notebooks

## Install

### Use GPU
 * To use gpu support
   * please install the right torch version manually <https://pytorch.org/get-started/locally/>
   * Or adjust setup.py with the machtching torch version
   * Be sure that you have a c compiler installed

### install with pip
 * `pip install .` without any torch backend. Manual installation of pytorch is needed
 * `pip install .[TORCH]` with the cuda version defined in 

## Usage:
 * Start training a model using  the gsap-ner-train script and a config file.
 * With an optional second parameter you can set the gpu cores to use (e.g. "0,2")
 * Example: `gsap-ner-train train/configs/gsap-scideberta.yaml`
 * Feel free to write your own config files
