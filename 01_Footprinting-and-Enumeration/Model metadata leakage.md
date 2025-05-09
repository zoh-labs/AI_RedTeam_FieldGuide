# 01_Footprinting-and-Enumeration: Model Metadata Leakage
> Tags: #ai-red-team #footprinting #enumeration #model-leakage #recon

## Overview

**Model metadata leakage** refers to the unintended exposure of technical details about an AI/ML model. Red teamers exploit these details to plan more precise attacks such as model extraction, data reconstruction, or prompt injection.

Common metadata includes:
- Model architecture (e.g., `GPT-2`, `ResNet50`)
- Parameter count (e.g., `345M`, `1.5B`)
- Frameworks used (e.g., `TensorFlow`, `PyTorch`)
- Training dataset details (e.g., `ImageNet`, internal data sources)
- Training configuration (e.g., epochs, learning rate)
- API versions and response formats
- Platform or infrastructure (e.g., `transformers==4.21.1`, `CUDA`, `ONNX`)

---

## Why It Matters

Exposed metadata can:
- Reveal known model vulnerabilities (CVE-level)
- Enable accurate model replication or stealing
- Reveal sensitive training sources or inference behavior
- Aid in adversarial example generation or prompt jailbreaks
- Undermine model confidentiality and IP

---

## Common Metadata Leakage Vectors

| Source                        | Description |
|------------------------------|-------------|
| **API Responses**            | Verbose error messages, leaked config in response JSON |
| **Public Model Cards**       | Hubs like HuggingFace, TensorFlow Hub |
| **GitHub Repositories**      | Training scripts, `.yaml`, `.ipynb`, model configs |
| **Swagger/OpenAPI Docs**     | Exposed schemas with parameter hints |
| **Frontend JS**              | Hardcoded model names, API keys |
| **CLI Tools**                | CLI output sometimes reveals backend model details |
| **Logs and Stack Traces**    | Debug mode leaks layer names, framework versions |

---

## Enumeration Tools & Commands

### Passive Recon

- **GitHub Dorking**
  ```bash
  site:github.com "model: gpt2" OR "torch.load" OR "config.json"
````

* **Google Dorks**

  ```bash
  site:huggingface.co inurl:model "trained on"
  site:pastebin.com "gpt2 config"
  ```

* **Shodan/Censys** (for API endpoints)

  ```bash
  shodan search "json model response"
  ```

### API Probing

* **cURL**

  ```bash
  curl -X POST https://target.com/api/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 10}'
  ```

* **Burp Suite** – Intercept and analyze API error messages, versions, or verbose logs.

* **Postman** – Test API endpoints for response structure, undocumented fields.

### Black Box Fingerprinting

* **Infer model size and type from output format**:

  * Consistency of sentence structure
  * Response latency (timing analysis)
  * Logit distributions (if exposed)

* **Response Analysis Script**:

  ```python
  import requests, time

  payload = {"prompt": "Test", "max_tokens": 10}
  start = time.time()
  r = requests.post("https://target.com/api/generate", json=payload)
  latency = time.time() - start
  print("Latency:", latency)
  print("Response:", r.json())
  ```

---

## Example: Leaky GPT Endpoint

**Request:**

```bash
curl -X POST https://example.com/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who is the CEO of OpenAI?", "max_tokens": 10}'
```

**Response:**

```json
{
  "model": "GPT-2-medium",
  "temperature": 0.7,
  "top_p": 0.9,
  "generated": "The current CEO of OpenAI is Sam",
  "version": "v1.3.2 (torch, cuda=0)"
}
```

**Leaked Info:**

* Model name: `GPT-2-medium`
* Backend: `torch`, `CUDA=0` (CPU-only)
* Temperature and `top_p`: used for output control
* Version: may allow exploit of known vulnerabilities

---

## Mitigation Strategies

| Countermeasure                    | Description                                                 |
| --------------------------------- | ----------------------------------------------------------- |
| **Limit API verbosity**           | Do not expose model names, frameworks, or configs           |
| **Strip version identifiers**     | Remove backend versioning in error or debug messages        |
| **Use proxy wrappers**            | Proxy requests to avoid revealing internal details          |
| **Output sanitization**           | Remove logits, confidence scores, and token IDs             |
| **Rate-limit & auth control**     | Prevent brute-force probing                                 |
| **Watermarking & fingerprinting** | Detect stolen/generated outputs downstream                  |
| **Security hardening**            | Remove `.ipynb`, `.yaml`, `.json` configs before deployment |
| **No debug mode in prod**         | Disable verbose logging, exceptions, tracebacks             |

---

## Red Team Checklist

* [ ] Enumerated `/docs`, `/api/info`, or Swagger endpoints
* [ ] Collected version strings or model names from response headers
* [ ] Reviewed GitHub/org pages for model config files
* [ ] Fuzzed API for undocumented parameters (`top_k`, `nucleus_sampling`)
* [ ] Probed timing differences in responses
* [ ] Extracted confidence scores, logits, or token IDs
* [ ] Flagged frontend JS with hardcoded backend identifiers
* [ ] Reported excessive verbosity in error messages

---

## References

* [HuggingFace Model Card Guide](https://huggingface.co/docs/hub/model-cards)
* [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
* [OpenAI Usage Policy](https://openai.com/policies/usage-policies)
* [Model Fingerprinting Techniques](https://arxiv.org/abs/2003.10595)

---

## File Location

This file is part of the AI Red Teaming Notes in the directory:

```

---

Let me know if you want a printable version, export to PDF/HTML, or the next topic filled in (e.g., **Membership Inference**, **API Enumeration**, etc.).

