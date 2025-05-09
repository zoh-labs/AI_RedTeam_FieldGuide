# Shadow Models and Training Attacks

**Tags:** #membership-inference #shadow-models #privacy #training-attacks #ai-red-teaming

---

## Overview

Shadow models and training attacks are a core technique in membership inference attacks (MIAs), where an attacker trains their own **"shadow" models** to mimic the behavior of a target model. These techniques are most effective in **black-box** settings, where internal model details are not known.

By observing the behavior of models trained on data with known membership, an attacker can train an **attack model** to infer membership of unknown data in the target model.

---

## Attack Phases

1. **Shadow Dataset Generation**
   Simulate a dataset from the same distribution as the target model’s training data.

2. **Shadow Model Training**
   Train one or more models (shadow models) on this synthetic or similar data.

3. **Attack Model Training**
   Using the shadow model(s), collect outputs for known members and non-members and train a binary classifier (the attack model) to distinguish between them.

4. **Membership Inference**
   Query the target model and feed the outputs to the trained attack model to infer if a data point was in the original training set.

---

## Shadow Model Architecture

* Ideally, replicate the architecture of the target model.
* If unknown, approximate it with a similar architecture based on domain knowledge (e.g., CNNs for images, transformers for NLP).

**Example Pseudocode:**

```python
# Train a shadow model on synthetic data
shadow_model = create_model()
shadow_model.fit(x_shadow_train, y_shadow_train)

# Collect model outputs for members and non-members
member_outputs = shadow_model.predict(x_shadow_train)
nonmember_outputs = shadow_model.predict(x_shadow_test)

# Train the attack model to classify member vs. non-member
attack_x = np.concatenate([member_outputs, nonmember_outputs])
attack_y = np.array([1]*len(member_outputs) + [0]*len(nonmember_outputs))

attack_model = LogisticRegression()
attack_model.fit(attack_x, attack_y)
```

---

## Tools and Libraries

* **Adversarial Robustness Toolbox (ART)**
  Provides built-in support for shadow models and membership inference attacks.
  📎 [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

* **CleverHans**
  MIT-developed tool supporting membership inference attack logic.
  📎 [https://github.com/cleverhans-lab/cleverhans](https://github.com/cleverhans-lab/cleverhans)

* **Sklearn / PyTorch / TensorFlow**
  Build shadow and attack models with standard ML frameworks.

---

## ART Example: Shadow Model Attack

```python
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from sklearn.ensemble import RandomForestClassifier

# Train target model
target_model = RandomForestClassifier()
target_model.fit(x_train, y_train)

# Wrap it for ART
art_model = SklearnClassifier(model=target_model)

# Launch attack
attack = MembershipInferenceBlackBox(art_model)
attack.fit(x_train, y_train)  # Train shadow models internally

# Infer membership
inferred = attack.infer(x_test, y_test)
print(inferred)
```

---

## Key Considerations

### Data Distribution Matching

* Shadow models require access to **data from the same distribution** as the target model's training data.
* In practice, adversaries may scrape data, synthesize data using generative models, or estimate the distribution from public sources.

### Number of Shadow Models

* Multiple shadow models improve generalization of the attack model.
* Vary training sets and architectures to simulate diverse model behavior.

---

## Enhancements

* **Use of GANs or synthetic data generators** to produce shadow training datasets that resemble the original training set.
* **Transfer learning** can improve attack performance when limited data is available.

---

## Defensive Measures

| Defense Technique        | Description                                                                       |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Differential Privacy** | Noise added during training makes model less sensitive to individual data points. |
| **Regularization**       | Penalizes overfitting, reducing leakage of training samples.                      |
| **Prediction Smoothing** | Reduce confidence scores to make outputs less indicative of membership.           |
| **Limiting API Access**  | Restrict query rate or output granularity (e.g., top-1 class only).               |

---

## Example Use Case

### Scenario:

* Target model: Commercial sentiment classifier hosted via API.
* Attacker:

  * Uses publicly scraped reviews to train shadow models.
  * Builds attack model to detect whether a specific customer review was used to train the commercial model.
  * Leverages high-confidence predictions to infer training membership.

---

## References

* **"Membership Inference Attacks Against Machine Learning Models"** – Shokri et al., 2017
  [https://arxiv.org/abs/1610.05820](https://arxiv.org/abs/1610.05820)

* **ART Documentation on Membership Inference**
  [https://adversarial-robustness-toolbox.readthedocs.io/](https://adversarial-robustness-toolbox.readthedocs.io/)

* **Shadow Model Implementation (Google Research)**
  [https://github.com/google/membership-inference](https://github.com/google/membership-inference)

---

## Conclusion

Shadow model attacks demonstrate how an adversary can simulate and exploit model behavior, even with black-box access, to perform powerful membership inference attacks. These attacks highlight the importance of privacy-preserving ML techniques, especially in models trained on sensitive or proprietary data.

---
