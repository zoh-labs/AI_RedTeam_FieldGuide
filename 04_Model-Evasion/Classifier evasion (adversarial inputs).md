# Classifier Evasion (Adversarial Inputs)

**Tags:** #adversarial #evasion #classifier #perturbation #deepfakes #AI-red-teaming #robustness

---

## Overview

Classifier evasion involves crafting **malicious inputs** that are misclassified by a target model. These inputs appear benign to humans but exploit weaknesses in the model’s decision boundaries.

Used heavily in:

* **Image, text, audio classifiers**
* **Content moderation**
* **Malware detection**
* **Biometrics / facial recognition**
* **Medical diagnostics**

---

## 1. Goals of Classifier Evasion

* **Mislead output**: Get a “safe” or incorrect label
* **Avoid detection**: Evade filters, firewalls, moderation
* **Trigger desired model behavior**: Without suspicion
* **Model fingerprinting**: Learn what inputs bypass defenses

---

## 2. Evasion Techniques by Modality

### a. Text-Based Evasion

Craft adversarial text inputs to fool NLP models.

#### Techniques:

* **Unicode injection / homoglyphs**: Replace characters with lookalikes
* **Typos / substitutions**: e.g. “b\@dword”
* **Synonym swaps**: Preserve semantics, break classifier
* **Invisible tokens**: Zero-width spaces, LTR/RTL markers
* **Prompt obfuscation**: Base64 or hex encoding

```python
# TextAttack example
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
attack = TextFoolerJin2019.build(model_wrapper)
```

---

### b. Image-Based Evasion

Create images that appear unchanged but are misclassified.

#### Techniques:

* **FGSM** (Fast Gradient Sign Method)
* **PGD** (Projected Gradient Descent)
* **Patch attacks**: Add visible or invisible patches
* **Color shifting**: Slight hue tweaks
* **JPEG compression artifacts**

```python
from art.attacks.evasion import FastGradientMethod
attacker = FastGradientMethod(estimator=model, eps=0.1)
x_adv = attacker.generate(x_test)
```

---

### c. Audio-Based Evasion

Trick speech or sound classifiers.

#### Techniques:

* **Adversarial perturbations**: Add noise below hearing threshold
* **TTS misuse**: Use synthetic voices with adversarial emphasis
* **Spectrogram alteration**
* **Ultrasonic command injection**

```python
# Adversarial audio example using Foolbox
import foolbox as fb
raw, _ = torchaudio.load("voice.wav")
adversarial = fb.attacks.L2FastGradientAttack()(model, raw, label)
```

---

## 3. Tools

| Tool                                     | Use                                             |
| ---------------------------------------- | ----------------------------------------------- |
| **TextAttack**                           | NLP adversarial generation                      |
| **Foolbox**                              | Image/audio adversarial attacks                 |
| **ART (Adversarial Robustness Toolbox)** | Multi-modal attacks & defenses                  |
| **CleverHans**                           | Classic white-box image attack library          |
| **DeepWordBug**                          | Character-level perturbations for NLP           |
| **AdvBox**                               | CV attacks including face recognition           |
| **SecEval**                              | Evaluates adversarial robustness of classifiers |

---

## 4. Code Example: FGSM on MNIST (PyTorch)

```python
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed = image + epsilon * sign_data_grad
    return torch.clamp(perturbed, 0, 1)

output = model(image)
loss = F.nll_loss(output, label)
model.zero_grad()
loss.backward()
data_grad = image.grad.data
adv_image = fgsm_attack(image, 0.1, data_grad)
```

---

## 5. Common Target Classifiers

* Hate speech filters
* Spam filters
* NSFW detectors
* Fake news classifiers
* Malware binary classifiers
* Medical diagnosis models (e.g., X-ray, pathology)

---

## 6. White-box vs. Black-box

| Type                 | Description                                        |
| -------------------- | -------------------------------------------------- |
| **White-box**        | Full access to model, weights, gradients           |
| **Black-box**        | Only output labels/scores available                |
| **Transfer attacks** | Train surrogate model and attack target indirectly |

---

## 7. Defense Mechanisms

| Defense                     | Description                                   |
| --------------------------- | --------------------------------------------- |
| **Adversarial training**    | Include adversarial examples in training set  |
| **Input preprocessing**     | JPEG compression, smoothing, spell correction |
| **Gradient masking**        | Obfuscate gradient signal (not foolproof)     |
| **Model ensemble**          | Use multiple classifiers                      |
| **Autoencoder filtering**   | Reconstruct input before classification       |
| **Confidence thresholding** | Reject low-confidence predictions             |

---

## 8. Detection Techniques

* High input perturbation norm
* Softmax confidence anomaly
* Activation vector drift
* Output entropy monitoring
* Ensemble disagreement

---

## 9. Attack Scenarios

| Scenario                 | Goal                                        |
| ------------------------ | ------------------------------------------- |
| Evade moderation filters | Bypass text/image classifiers               |
| Fool facial recognition  | Unlock phone with adversarial glasses       |
| Spam classifier bypass   | Deliver phishing messages                   |
| Disguise malware         | Evade static ML antivirus                   |
| Fool medical AI          | Trick diagnostic system into false negative |

---

## 10. References & Resources

* [TextAttack GitHub](https://github.com/QData/TextAttack)
* [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* [Foolbox](https://github.com/bethgelab/foolbox)
* [CleverHans](https://github.com/cleverhans-lab/cleverhans)
* "Explaining and Harnessing Adversarial Examples" – Goodfellow et al. ([https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572))

---
