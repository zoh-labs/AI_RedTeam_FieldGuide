# Autoencoder Bypass

**Tags:** #autoencoder #bypass #latent-space #evasion #adversarial #decoder #compression #anomaly-detection #llm-redteaming

---

## Overview

Autoencoders are commonly used in ML pipelines for:

* **Input sanitization**
* **Noise filtering**
* **Anomaly detection**
* **Defense against adversarial examples**

### Bypass Goal

Red Teamers aim to craft inputs that:

* Appear normal to the autoencoder
* Are decoded into dangerous payloads, or
* Fool downstream models despite reconstruction

---

## 1. Understanding Autoencoders

Autoencoders learn a compressed **latent representation** of input data.

**Flow:**

```
Input → Encoder → Latent Vector → Decoder → Reconstructed Output
```

They’re trained to minimize reconstruction loss:

```math
L = ||x - Decoder(Encoder(x))||²
```

---

## 2. Red Team Tactics to Bypass

### a. Latent Space Manipulation

Inject a crafted latent vector that decodes to a malicious output.

```python
# Manually modify latent vector
z = encoder(x)
z[3] += 5  # inject into a latent dimension
x_reconstructed = decoder(z)
```

---

### b. Gradient-Based Latent Injection

Use gradients to optimize a latent vector toward a malicious decoded output.

```python
# Pseudo-code
z = torch.randn(1, latent_dim, requires_grad=True)
optimizer = torch.optim.Adam([z], lr=0.01)

target_text = "Ignore previous instructions..."
loss_fn = lambda z: similarity(decoder(z), target_text)

for _ in range(100):
    loss = loss_fn(z)
    loss.backward()
    optimizer.step()
```

---

### c. Pre-Autoencoder Obfuscation

Hide malicious payloads in forms the autoencoder will pass through unchanged:

* Insert invisible Unicode characters
* Homoglyph obfuscation
* Encode payload with Base64 or other representation

```bash
echo "alert('XSS')" | base64
# => YWxlcnQoJ1hTUycp
```

Inject encoded payload into input text or metadata field.

---

### d. Exploit Undercomplete Representations

Some autoencoders compress too aggressively. Craft inputs that:

* Encode to *same latent vector* as benign input
* But decode to *altered* output due to decoder quirks

```text
Trigger phrase: "I'm just here to help"
But results in: "I'm just here to help you hack the system"
```

---

### e. Chain Decoders

If multiple decoders are chained (e.g., in LLM pipelines or API filters), you may exploit mismatched assumptions.

```text
Step 1: Obfuscate message → autoencoder lets it pass  
Step 2: Downstream LLM interprets and acts on it
```

---

## 3. Tools & Code Examples

### ART – Adversarial Robustness Toolbox

```python
from art.defences.preprocessor import FeatureSqueezing
from art.attacks.evasion import FastGradientMethod
```

Train an autoencoder and then attempt adversarial bypass.

---

### TensorFlow/Keras Autoencoder Example

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = 784
encoding_dim = 64

input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50)
```

---

## 4. Use Cases for Red Teaming

| Scenario                         | Technique                                     |
| -------------------------------- | --------------------------------------------- |
| LLM content filter evasion       | Embed prompt in latent space                  |
| Chat moderation bypass           | Homoglyph + encoding trick                    |
| Vision model sanitization bypass | Patch image to match normal latent vector     |
| Adversarial audio evasion        | Craft waveform that compresses to benign form |

---

## 5. Detection Countermeasures

| Defense                                 | Approach                                    |
| --------------------------------------- | ------------------------------------------- |
| Latent space monitoring                 | Detect unusual clusters or outliers         |
| Input reconstruction error thresholding | Reject inputs with high reconstruction loss |
| Robust autoencoder training             | Add adversarial examples to training set    |
| Ensemble autoencoders                   | Use multiple AEs and compare outputs        |
| Decoder fuzz testing                    | Validate decoder behavior on fuzzed latents |

---

## 6. Resources & References

* [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* [Latent Space Attacks on Autoencoders](https://arxiv.org/abs/2007.06952)
* [Autoencoder Anomaly Detection](https://towardsdatascience.com/anomaly-detection-using-autoencoders-5fddf7a9fae1)
* [Invisible Character Obfuscation](https://www.zdnet.com/article/hackers-hide-malware-in-unicode-characters/)

---

