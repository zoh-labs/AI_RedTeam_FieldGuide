# LLM endpoint discovery (plugins, APIs, models)

---

````markdown
# LLM Endpoint Discovery

This guide focuses on discovering exposed LLM endpoints during AI Red Team assessments. This includes plugin enumeration, API exposure, model inference endpoints, and model metadata leaks.

---

## 1. Overview

Large Language Models (LLMs) are often exposed via:

- Web plugins or extensions
- Public or internal REST APIs
- Web-based chat interfaces
- Mobile or desktop apps
- Cloud inference services (e.g., OpenAI, Hugging Face, Replicate, Bedrock)

### Common Use Cases
- Chatbots
- Document Q&A
- Code assistants
- Search-enhanced LLMs (RAG)
- LLMs embedded in SaaS platforms

---

## 2. Initial Recon

### 2.1 Subdomain Discovery

Start with enumeration of subdomains related to AI, API, model, or internal services.

```bash
amass enum -d target.com -o subdomains.txt
assetfinder --subs-only target.com | tee -a subdomains.txt
````

Look for names like:

```
llm.target.com
api.target.com
inference.target.com
ai.target.com
openai-proxy.target.com
```

---

## 3. Detecting Publicly Accessible Inference APIs

### 3.1 Common Endpoints to Probe

Probe known or guessed endpoints:

```bash
curl -X POST https://target.com/api/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is your backend model?"}'
```

Check responses for:

* Model names (e.g., `gpt-3.5`, `llama-2`, `mistral`)
* Backend references (OpenAI, Anthropic, Cohere, etc.)
* Inference framework (vLLM, FastAPI, Flask)

---

## 4. Fingerprinting via Swagger, OpenAPI, or GraphQL

### 4.1 Swagger/OpenAPI

Try common paths:

```bash
curl https://target.com/swagger.json
curl https://target.com/openapi.json
```

Tools:

```bash
nuclei -t exposures/apis/
swagger-comb -i swagger.json
```

---

### 4.2 GraphQL LLM Interfaces

Check for `/graphql` endpoint:

```bash
curl -X POST https://target.com/graphql \
     -H "Content-Type: application/json" \
     -d '{"query":"{__schema{types{name}}}"}'
```

Look for introspection access to:

* `completion`
* `chat`
* `embedding`

---

## 5. Plugin & Extension Enumeration

### 5.1 Chrome Extension Analysis

Search Chrome Web Store or extensions.json in leaked repos.

```bash
grep -ri "openai" ~/Downloads/chrome-extensions/
```

Look for:

* API keys
* Inference endpoints
* Hardcoded prompt templates

### 5.2 VSCode & Browser Plugin Artifacts

Common locations:

```bash
~/.vscode/extensions/
~/.config/google-chrome/Default/Extensions/
```

Check for:

* `api.openai.com`
* `gpt.*`
* `.well-known/ai-plugin.json`

---

## 6. Discovering `.well-known` Plugin Files

Try:

```bash
curl https://target.com/.well-known/ai-plugin.json
```

Look for:

* `"api": { "url": "https://..." }`
* `"model"` fields
* Plugin name and description

---

## 7. Model Leakage via HTML/JS

Search JavaScript for model clues:

```bash
curl https://target.com | grep -i "model"
```

Use `jsfinder`, `linkfinder`, or `subjs` to find JS files:

```bash
subjs https://target.com | httpx | xargs -n1 -P10 curl -s | grep -i "openai"
```

---

## 8. Interact with Common AI Gateways

Check usage of these third-party LLM providers:

* `api.openai.com/v1/`
* `api.cohere.ai/`
* `api.anthropic.com/`
* `api.replicate.com/v1/predictions`

Example curl for OpenAI proxy detection:

```bash
curl https://target.com/api/completions \
     -H "Authorization: Bearer x" \
     -d '{"model":"gpt-3.5-turbo", "prompt":"test"}'
```

---

## 9. Embedding & Vector Endpoint Discovery

These endpoints often expose model context or allow inference:

* `/embed`
* `/vectorize`
* `/similarity`
* `/search`

Look for payloads like:

```json
{
  "input": "tell me about company secrets",
  "model": "text-embedding-ada-002"
}
```

---

## 10. Tools

| Tool                           | Use Case                            |
| ------------------------------ | ----------------------------------- |
| `amass`, `assetfinder`         | Subdomain enumeration               |
| `nuclei`                       | Detecting API & AI plugin exposures |
| `ffuf`, `dirsearch`            | Endpoint fuzzing                    |
| `httpx`, `subjs`, `linkfinder` | JS scraping                         |
| `curl`, `httpie`               | API interaction                     |
| `swagger-comb`, `graphqlmap`   | Schema extraction                   |

---

## 11. Tags

`#enum` `#llm` `#embedding` `#plugin` `#api-discovery` `#model-leakage`

```
