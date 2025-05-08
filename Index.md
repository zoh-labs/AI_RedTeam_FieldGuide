AI-RedTeam/
├── 00_Index.md
│   └─ Table of contents with tags & references

├── 01_Footprinting-and-Enumeration.md
│   ├─ LLM endpoint discovery (plugins, APIs, models)
│   ├─ Model metadata leakage
│   ├─ Dataset discovery (open and embedded)
│   ├─ Embedding enumeration and distance metrics
│   └─ Tags: #enum #llm #embedding

├── 02_Prompt-Injection.md
│   ├─ Direct and indirect injection
│   ├─ Nested injection and prompt chaining
│   ├─ Role hijacking, few-shot overrides
│   ├─ HTML/script prompt payloads
│   └─ Tags: #prompt-injection #jailbreak #llm

├── 03_Jailbreaks.md
│   ├─ DAN, Grandma, Developer Mode, etc.
│   ├─ Emoji, unicode, punctuation encoding tricks
│   ├─ Long-token or repetition-based jailbreaks
│   └─ Tags: #jailbreak #prompt-injection #llm

├── 04_Model-Evasion.md
│   ├─ Classifier evasion (adversarial inputs)
│   ├─ Text/image perturbation techniques
│   ├─ Autoencoder bypass
│   └─ Tags: #evasion #adversarial #ml

├── 05_Model-Inversion.md
│   ├─ Gradient-based inversion (ML)
│   ├─ Prompt-based reconstruction (LLM)
│   └─ Tags: #inversion #privacy #ml #llm

├── 06_Membership-Inference.md
│   ├─ White-box and black-box inference
│   ├─ Shadow models and training attacks
│   └─ Tags: #membership-inference #privacy #ml

├── 07_Data-Poisoning.md
│   ├─ Label flipping, backdoor triggers
│   ├─ Pre-training prompt pollution
│   ├─ Embedding corruption and Trojaning
│   └─ Tags: #poisoning #training #supplychain

├── 08_Embedding-Attacks.md
│   ├─ Insertion in semantic search
│   ├─ Injection via embeddings
│   ├─ Manipulation of retrieval results (RAG poisoning)
│   └─ Tags: #embedding #rag #vector-db

├── 09_Model-Extraction.md
│   ├─ Output truncation sampling
│   ├─ API fingerprinting
│   ├─ Copycat model training from black-box output
│   └─ Tags: #model-extraction #recon #llm

├── 10_Supply-Chain-Attacks.md
│   ├─ Model weight tampering
│   ├─ Compromised pip packages or HuggingFace repos
│   ├─ External plugin abuse (e.g., LangChain, LlamaIndex)
│   └─ Tags: #supplychain #plugin #model

├── 11_Logging-and-Detection-Evasion.md
│   ├─ Token-level evasion
│   ├─ Stopword obfuscation
│   ├─ Bypassing filters via encoding
│   └─ Tags: #evasion #logging #monitoring

├── 12_Agent-Manipulation.md
│   ├─ LangChain & AutoGPT abuse scenarios
│   ├─ Instruction leakage and memory manipulation
│   └─ Tags: #agents #autogpt #langchain #memory

├── 13_Attack-Chains.md
│   ├─ Prompt injection → memory poisoning → exfiltration
│   ├─ Embedding injection → vector recall → jailbreak
│   └─ Tags: #chaining #multi-step #offense

├── 14_Tools.md
│   ├─ ART (Adversarial Robustness Toolbox)
│   ├─ TextAttack, CleverHans, LLMGuard
│   ├─ RedTeaming LLMs (Meta), LLM-Fuzzer, langfuse
│   └─ Tags: #tools #art #textattack #cleverhans #llmguard

├── 15_Defense-and-Mitigation.md
│   ├─ Input validation / sanitization
│   ├─ Prompt hardening & contextual windows
│   ├─ Output filtering & anomaly detection
│   ├─ AI-specific WAFs and monitoring (e.g., LLMProtect)
│   └─ Tags: #defense #mitigation #hardening

└── 16_References-and-Reading.md
    ├─ OWASP LLM Top 10
    ├─ MITRE ATLAS mapping
    ├─ Real-world case studies
    └─ Tags: #owasp #mitre #case-study
