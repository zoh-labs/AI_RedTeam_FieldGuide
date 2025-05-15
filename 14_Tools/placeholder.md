
* **Type of attack**
* **Use case (when/why to use it)**
* **Brief explanation of what the command does**

---

# Common Command-Line Snippets for Adversarial Attacks

This cheatsheet includes CLI or script-style commands for launching adversarial attacks using different frameworks like ART, CleverHans, Foolbox, TextAttack, and PyTorch-based scripts. Each snippet notes the type of attack, when it's useful, and what the command does.

---

## Adversarial Robustness Toolbox (ART)

### 1. FGSM Attack on MNIST

**Type:** Evasion attack – untargeted
**Use case:** Simple, fast baseline attack to test model robustness on image classification

```bash
python3 -m art.estimators.classification.run_attack --dataset MNIST --attack FGSM
```

This runs a Fast Gradient Sign Method (FGSM) attack using ART’s built-in runner against a model trained on MNIST.

---

### 2. Evaluate Accuracy Under Attack

**Type:** Evaluation phase for adversarial robustness
**Use case:** To measure how vulnerable a model is to a specific attack

```bash
python3 -m art.estimators.classification.evaluate --model saved_model.h5 --attack FGSM
```

Evaluates a pre-trained Keras model against adversarial examples generated using FGSM.

---

### 3. Save Adversarial Examples

**Type:** Data generation (for later use)
**Use case:** Pre-generating adversarial samples for offline testing

```bash
python3 -m art.scripts.generate --attack FGSM --save adversarials.npy
```

Runs an FGSM attack and saves the adversarial images in `.npy` format.

---

## CleverHans

### 4. FGSM on MNIST

**Type:** Evasion attack
**Use case:** Common baseline for evasion tests using gradient-based perturbations

```bash
python3 cleverhans_tutorials/mnist_tutorial_fgsm.py
```

Runs FGSM on a standard MNIST model using CleverHans’ tutorial script.

---

### 5. Carlini-Wagner L2 Attack

**Type:** Evasion attack – targeted
**Use case:** Stronger, optimization-based attack, useful for evaluating high-security models

```bash
python3 cleverhans_tutorials/mnist_tutorial_cw.py
```

Launches a CW L2 attack that finds small but effective perturbations to change classification.

---

### 6. Projected Gradient Descent (PGD)

**Type:** Iterative evasion attack
**Use case:** Stronger than FGSM, often used to test against robust models

```bash
python3 cleverhans_tutorials/mnist_tutorial_pgd.py
```

Performs a PGD attack, which repeatedly applies gradient-based updates within a bound.

---

## Foolbox

### 7. FGSM using Foolbox

**Type:** Evasion attack
**Use case:** Quick perturbation-based attack to check model sensitivity

```bash
python3 foolbox_examples/fgsm_mnist.py
```

Executes an FGSM attack against an MNIST model using Foolbox.

---

### 8. LinfPGD Attack

**Type:** Iterative evasion attack
**Use case:** Used to test certified robustness claims of models

```bash
python3 foolbox_examples/linf_pgd_attack.py --model model.pt
```

Runs PGD under the Linf norm constraint on a PyTorch model.

---

### 9. Save Adversarial Samples

**Type:** Data preparation
**Use case:** Store adversarial outputs for manual inspection or secondary experiments

```bash
python3 foolbox_examples/save_adversarials.py --out ./results/adversarials/
```

Saves adversarial examples from a previous run to a specified directory.

---

## TextAttack (for NLP)

### 10. TextFooler Attack

**Type:** Evasion (text classification)
**Use case:** Attack NLP models using synonym replacements while preserving semantics

```bash
textattack attack --model bert-base-uncased-imdb --recipe textfooler
```

Launches a TextFooler attack on a pretrained BERT model using the IMDB dataset.

---

### 11. Attack Custom HuggingFace Model

**Type:** Evasion (NLP)
**Use case:** Evaluating fine-tuned or custom language models

```bash
textattack attack --model-from-huggingface ./my_model --dataset imdb --recipe bert-attack
```

Runs a BERT-Attack on a local HuggingFace model against IMDB data.

---

### 12. Save NLP Attack Output

**Type:** Logging
**Use case:** Store attack outcomes for review and reporting

```bash
textattack attack --model roberta-base-sst2 --recipe pruthi --output-dir ./logs/
```

Saves full attack logs, predictions, and adversarial texts to a local folder.

---

## Torchattacks (manual/scripted)

### 13. FGSM with Torchattacks

**Type:** Evasion
**Use case:** Simple and fast benchmark attack for PyTorch-based models

```bash
python3 attacks/fgsm_mnist.py --eps 0.3 --model saved_model.pt
```

Runs an FGSM attack with an epsilon of 0.3 on a saved PyTorch MNIST model.

---

### 14. PGD Attack

**Type:** Evasion, iterative
**Use case:** Stronger attack for evaluating robust models

```bash
python3 attacks/pgd_attack.py --eps 0.3 --alpha 0.01 --steps 40
```

Applies PGD with a defined epsilon and step size over 40 iterations.

---

## Custom Scripts (General Pattern)

### 15. Run Custom FGSM Attack

**Type:** Evasion
**Use case:** When using your own training/testing pipeline

```bash
python3 run_fgsm.py --model model.pth --data mnist --eps 0.25
```

Custom wrapper script running FGSM on a specific model and dataset.

---

### 16. Evaluate Robustness from Adversarials

**Type:** Evaluation
**Use case:** Measuring accuracy of a model under attack

```bash
python3 evaluate_adversarial.py --attack pgd --dataset cifar10 --output results.json
```

Runs evaluation of adversarial samples on CIFAR-10 and outputs metrics to JSON.

---

## Utility & Setup Commands

### 17. Export Model to ONNX

**Type:** Model conversion
**Use case:** To make models portable across platforms/tools

```bash
python3 export_model.py --input model.pth --output model.onnx
```

Converts a PyTorch model to ONNX format.

---

### 18. Convert Keras Model to TFLite

**Type:** Model conversion
**Use case:** For mobile model testing or lightweight inference

```bash
tflite_convert --saved_model_dir=model_dir --output_file=model.tflite
```

Converts a saved Keras model to TensorFlow Lite format.

---

### 19. Setup Virtual Environment

**Type:** Environment setup
**Use case:** Clean Python environment for testing adversarial tools

```bash
python3.10 -m venv advlab-env && source advlab-env/bin/activate
pip install -r requirements.txt
```

Creates and activates a virtual environment, then installs dependencies.

---

### 20. Clean Adversarial Output

**Type:** Maintenance
**Use case:** Free up space and reset environment

```bash
rm -rf ./results/adversarials/*
```

Deletes stored adversarial examples from the output directory.

---
