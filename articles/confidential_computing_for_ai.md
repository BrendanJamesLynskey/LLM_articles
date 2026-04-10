# Confidential Computing for AI: Trusted Execution for LLMs

*April 2026 • Technical Report*

## 1. Introduction

When an enterprise sends its data to a hosted LLM provider, it must implicitly trust that provider with the data's confidentiality and integrity. The provider's employees, the cloud infrastructure they run on, the hypervisor, the operating system, the model serving software, and the GPUs themselves all have technical access to the prompts and responses flowing through them. For most workloads, this trust is acceptable — the user has signed a contract, the provider has security audits, and the actual risk of malicious access is low. For some workloads, it is not. Healthcare data, financial records, classified government information, attorney-client communications, and trade secrets all carry legal or commercial requirements that the provider not be able to access them at all.

Confidential computing is the cryptographic and hardware-based technology stack that addresses this gap. It allows code to run on infrastructure controlled by a third party while ensuring that no one — not even the infrastructure provider, the cloud's root user, or someone with physical access to the machine — can read the code's data or interfere with its execution. The data is encrypted in memory, the execution happens inside a hardware-enforced enclave, and remote attestation provides cryptographic proof that the right code is running on real hardware before any sensitive data is exposed to it.

For most of its history, confidential computing has been a niche technology used for narrow security-sensitive applications. The arrival of LLMs has given it a much broader audience. Suddenly there is widespread demand for "I want to use a frontier LLM but the provider must not see my data," and the only way to satisfy this demand is some form of trusted execution. NVIDIA's Hopper and Blackwell GPUs include confidential computing features specifically designed for AI workloads, and major cloud providers have started offering confidential AI services built on top of them.

This report examines how confidential computing works, how it has been adapted for GPU-accelerated AI, and what the practical deployment and trust models look like.

## 2. Trusted Execution Environments

### 2.1 The TEE Concept

A Trusted Execution Environment (TEE) is a hardware-isolated region of a processor where code can run with strong guarantees about confidentiality and integrity. The hardware enforces that:

- Code outside the TEE cannot read memory inside the TEE
- Code inside the TEE cannot be modified by code outside the TEE
- The contents of TEE memory are encrypted before being written to RAM, so they cannot be read even with physical access to memory chips
- The CPU itself cryptographically attests to which code is running inside the TEE, allowing remote parties to verify they are talking to a legitimate enclave running their expected software

The hardware enforcement is the critical part. Conventional security relies on software boundaries — the operating system, the hypervisor, the container runtime — that can be bypassed by anyone with sufficient privilege. TEEs are designed so that even the processor's root user (typically called Ring -1 or higher) cannot peek inside the TEE's memory.

### 2.2 CPU TEE Implementations

The major CPU vendors have all developed TEE technologies:

- **Intel SGX (Software Guard Extensions)**: The first widely-deployed CPU TEE, introduced in Skylake processors in 2015. SGX creates "enclaves" that isolate small portions of an application's memory. SGX has been the standard for confidential computing in research and many production systems, though it has known side-channel vulnerabilities and limited memory capacity.
- **Intel TDX (Trust Domain Extensions)**: A newer Intel TEE that operates at the VM level rather than the application level. A TDX-protected VM can run an entire operating system inside the TEE, which is more practical than SGX's small enclaves. TDX is the basis for most newer confidential VM offerings.
- **AMD SEV-SNP (Secure Encrypted Virtualization with Secure Nested Paging)**: AMD's VM-level TEE, supporting full encrypted VMs with attestation. SEV-SNP and Intel TDX are roughly comparable in capability.
- **ARM CCA (Confidential Compute Architecture)**: ARM's equivalent for ARMv9 server chips, used in some AWS Graviton processors.

For LLM workloads, the VM-level TEEs (TDX, SEV-SNP) are dominant because they allow standard inference stacks to run inside the protected environment with minimal modification.

### 2.3 Remote Attestation

A TEE is only useful if the user can verify that the right code is running inside it. Remote attestation is the protocol that provides this verification. The TEE's hardware generates a signed measurement of the code currently running inside it, including the boot loader, the OS, and the application binaries. A remote verifier (the user, or a trusted attestation service) checks this measurement against expected values and verifies the signature using public keys rooted in the CPU vendor's certificate authority.

Once attestation succeeds, the user knows two things: the code running in the TEE is exactly the code they expect (no malicious modifications), and the TEE is genuine (running on real hardware, not a software simulator). They can then transmit sensitive data to the TEE with confidence that only the verified code will be able to read it.

The attestation flow typically looks like:
1. User establishes a TLS connection to the TEE
2. TEE generates an attestation report containing measurements and a fresh public key
3. User verifies the report with the CPU vendor's attestation service
4. User establishes a session key with the TEE using the verified public key
5. User transmits sensitive data over the encrypted session

This flow happens on every connection, so the user is always verifying the TEE's identity before sending data.

## 3. The GPU Problem

### 3.1 Why CPU TEEs Are Not Enough

CPU TEEs provide confidentiality for data processed on the CPU. But AI inference happens on GPUs, and GPU memory has historically been outside the TEE boundary. A naive confidential AI deployment that runs LLM inference on a TDX VM would still expose the model weights and KV cache to the GPU, which is accessible via DMA from outside the TEE. The encryption stops at the PCIe bus, and anything the GPU touches is visible to anyone who can access the GPU.

This is a fatal flaw for confidential AI. The vast majority of the sensitive data — the prompts being processed by the model, the model's intermediate activations, the generated outputs — exists in GPU memory during inference. If the GPU is outside the TEE, the entire confidentiality story collapses.

The solution is to extend the TEE boundary to include the GPU. NVIDIA's Hopper architecture introduced the first widely-available implementation of this idea, called GPU Confidential Computing or NVIDIA Confidential Computing.

### 3.2 NVIDIA Confidential Computing on Hopper and Blackwell

The Hopper H100 was the first NVIDIA GPU to support confidential computing as a first-class feature. The implementation includes several pieces:

- **Encrypted PCIe transfers**: All data flowing between CPU and GPU is encrypted in transit, so a malicious DMA from a third party cannot read it.
- **Encrypted GPU memory**: Data stored in HBM is encrypted using a key that is generated on the GPU and never leaves it.
- **GPU-side attestation**: The GPU produces its own attestation reports that can be verified alongside the CPU's TDX or SEV-SNP attestation.
- **Secure mode boot**: The GPU firmware verifies its own integrity before the GPU enters confidential computing mode, preventing tampering at the firmware level.

A typical confidential AI deployment looks like: TDX (or SEV-SNP) protects the CPU side, NVIDIA Confidential Computing protects the GPU side, and the two attestation chains are combined into a single verifiable claim that the entire CPU+GPU compute pipeline is running approved code on genuine hardware.

Blackwell extends this with improved performance (the encryption overhead on Hopper was 5-15% for typical inference workloads; Blackwell reduces this to 1-3%) and better tooling for managing attestation in multi-GPU and multi-node scenarios.

### 3.3 The Performance Cost

Encryption is not free. On Hopper, NVIDIA Confidential Computing typically adds 5-15% to inference latency, depending on the model size and batch configuration. The cost comes from encrypting and decrypting data on the PCIe bus and from the additional crypto operations needed to maintain the confidentiality guarantees.

For workloads that can tolerate this overhead — and most can — the trade-off is favorable. A 10% inference cost increase in exchange for cryptographic guarantees about data confidentiality is a good deal for sensitive workloads. For workloads that cannot tolerate any overhead, the choice is to either run the inference outside a TEE (and accept the trust assumptions of normal cloud computing) or to use a different approach like dedicated hardware.

## 4. Confidential Inference Deployments

### 4.1 The Trust Model

A confidential AI deployment establishes a specific trust model. The user trusts:

- The CPU and GPU vendors (Intel/AMD/NVIDIA) to have built the TEE features correctly
- The TEE attestation infrastructure (typically operated by the chip vendors) to correctly issue attestation certificates
- The code that the user has approved to run inside the TEE (the inference server, the model, the OS, the boot loader)

The user does not need to trust:

- The cloud provider's employees
- The cloud provider's hypervisor or host OS
- Anyone with physical access to the cloud datacenter
- Other tenants on the same physical hardware
- Network operators between the user and the TEE

This is a significant reduction in the trusted computing base compared to conventional cloud computing, where the entire cloud provider's infrastructure must be trusted.

### 4.2 Cloud Provider Offerings

The major cloud providers all offer confidential AI services as of 2025:

- **Microsoft Azure**: Azure Confidential Computing extends to NVIDIA H100 and H200 GPUs with TDX-protected confidential VMs. Azure's confidential AI offerings include hosted models and BYO models running inside these VMs.
- **Google Cloud**: Confidential Space provides TDX-based confidential VMs with NVIDIA GPU support. Google has integrated this with their Vertex AI platform for selected use cases.
- **Amazon Web Services**: AWS Nitro Enclaves provide CPU-side confidential computing, but AWS has been slower to add GPU support. The recent G6e instances include confidential computing capabilities for L40S GPUs, with broader support expected in upcoming instance types.
- **Oracle Cloud, IBM Cloud**: Both offer confidential VM products with varying levels of GPU support.

The hyperscalers' offerings differ in implementation details (which CPU TEE they use, which GPU models are supported, how they manage attestation) but provide broadly comparable capabilities. The choice between them is usually driven by which cloud the rest of the customer's infrastructure runs on.

### 4.3 Confidential AI as a Service

Several specialized companies offer confidential AI as a complete service, abstracting away the TEE configuration and providing a normal-looking API. The user submits prompts via an API, and the provider's backend runs them through an LLM inside a TEE without ever having access to the plaintext content. Examples include:

- **Edgeless Systems** (Continuum AI): Confidential AI inference with attestation, currently supporting open-source models running in Intel TDX environments with NVIDIA GPU confidential computing.
- **Phala Network**: A blockchain-integrated confidential computing platform that offers AI inference with on-chain attestation receipts.
- **Apple Private Cloud Compute**: Apple's confidential computing platform for routing certain Apple Intelligence requests off-device. Uses custom Apple Silicon with confidential computing features and verified system images.

Apple Private Cloud Compute is particularly noteworthy because it represents the first mass-market deployment of confidential AI to consumer users. When an iPhone needs to process a request that exceeds on-device model capabilities, it can route the request to Apple's PCC, which runs the inference inside a verified enclave and returns the result. The user is told via the OS that PCC is being used, and security researchers can audit the PCC software stack independently.

## 5. Confidential Training

Training is harder than inference for confidential computing, for several reasons. Training is much more compute-intensive, so the encryption overhead matters more. Training spans many GPUs and many nodes, requiring confidential communication across the entire cluster. Training datasets are much larger than typical inference inputs, requiring secure data ingestion at scale.

The current state of confidential training is "nascent." A few research papers and limited commercial offerings exist, but the practical deployments are rare. The most likely use case is collaborative training where multiple organizations contribute data without revealing it to each other (federated training with TEE backstops), but this is mostly research as of 2025.

Inference, by contrast, is mature enough for real production use. Most discussion of "confidential AI" implicitly means confidential inference.

## 6. Side Channels and Limitations

TEEs are not impenetrable. They protect against the threat models they were designed for (curious or malicious infrastructure operators, software-level attacks from outside the TEE) but they have known weaknesses:

### 6.1 Side-Channel Attacks

Side-channel attacks exploit information leakage through indirect channels — timing, cache behavior, power consumption, electromagnetic emanations — rather than directly reading TEE memory. SGX in particular has been vulnerable to a long series of side-channel attacks (Foreshadow, Plundervolt, ZombieLoad, AEPIC Leak, and others), each of which required microcode updates and partially compromised the security guarantees.

TDX and SEV-SNP have fewer published side-channel vulnerabilities, but they are also newer and less heavily studied. NVIDIA's GPU confidential computing is even newer, and the side-channel attack surface on GPUs is poorly understood. A motivated attacker with deep expertise and physical access to the hardware may be able to extract some information through side channels, though doing so at scale is difficult.

For most threat models, side channels are acceptable risks because the attacks are slow, expensive, and produce limited information. For threat models involving nation-state adversaries or very high-value targets, side channels may matter more.

### 6.2 Vendor Trust

Confidential computing requires trusting the chip vendor's implementation of the TEE features. If Intel, AMD, or NVIDIA introduced a backdoor in their TEE silicon — intentionally or through a bug — the entire confidentiality story would fail. There is no way for users to verify the silicon implementation independently.

This is an unavoidable trust dependency. The mitigations are limited: users can choose vendors they consider trustworthy, monitor for published vulnerabilities, and (for the most sensitive use cases) use multiple TEE implementations from different vendors so that compromising any single vendor is insufficient.

### 6.3 Code Trustworthiness

Attestation verifies that the right code is running, but the user must still trust that the code itself is benign and correctly implemented. A confidentially-running inference server with a bug that leaks data to logs is just as compromising as a non-confidential one. The TEE protects against external attackers but not against the user's own software bugs.

The mitigation is to keep the TEE-resident code small and auditable. Open-source inference servers running in confidential VMs are easier to trust than closed-source ones. Apple's PCC publishes its software images for security researchers to audit precisely for this reason.

## 7. The Regulatory Driver

Much of the demand for confidential AI comes from regulated industries. Healthcare (HIPAA), finance (PCI-DSS, SOX), legal services (attorney-client privilege), defense (classified data handling), and government contractors all have legal requirements that limit how they can use third-party services. A hospital cannot send patient records to a cloud LLM API without a Business Associate Agreement and substantial compliance infrastructure. A bank cannot send customer financial data to an LLM provider without satisfying its regulators. A defense contractor cannot use an LLM for classified work without a FedRAMP High authorization.

Confidential computing offers a path to compliance for these use cases. By providing cryptographic guarantees that the LLM provider cannot access the data, it changes the legal analysis. The data is still being processed in the cloud, but it is processed in a way that the cloud provider cannot read, which can satisfy many of the requirements that would otherwise rule out cloud LLM use.

The regulatory acceptance of confidential computing is still being established. Some regulators are explicit that TEE-based protections are sufficient; others are still working through the implications. The trend is toward acceptance, but the pace varies by jurisdiction and industry.

## 8. The Open Questions

Several open questions remain about confidential AI:

**How much overhead is acceptable?** The current 1-15% overhead on GPU inference is acceptable for many workloads but may be a barrier for cost-sensitive deployments. Future hardware generations should reduce this further, but the floor is unlikely to be zero.

**How do you handle multi-GPU and multi-node confidential computing at scale?** The attestation chains and key management get more complex as the deployment scales. Current implementations handle small confidential clusters well; large-scale confidential training remains a research problem.

**How do you balance confidentiality with observability?** A confidentially-running model is a black box from the operator's perspective. Standard observability tools cannot inspect the data flowing through it. New observability approaches that respect TEE boundaries — recording metadata without recording content — are needed.

**What about provider trust at the model level?** Confidential computing protects against the cloud infrastructure provider, but the model itself is provided by another party. Trusting that the model has not been backdoored to leak data through its outputs requires a different set of mitigations (model audits, output filtering, bounded responses).

## 9. The Bigger Picture

Confidential AI is one of the few technologies that can change the trust model of LLM deployments. Without it, sensitive data cannot leave the user's premises, which excludes most cloud LLM offerings. With it, sensitive data can be processed in the cloud with cryptographic confidentiality guarantees, opening up the entire ecosystem of hosted models to regulated workloads.

The technology is mature enough for production deployment in 2025. The cost overhead is acceptable, the cloud providers offer credible products, and the compliance story is increasingly well-understood by regulators. Adoption is still limited — most LLM use does not require confidential computing — but the use cases that do require it are growing as more organizations adopt LLMs for sensitive tasks.

## 10. Conclusion

Confidential computing for AI is the bridge between two seemingly contradictory requirements: the desire to use the most powerful frontier LLMs and the requirement to keep sensitive data away from third parties. By using hardware-based trusted execution environments that extend across both CPUs and GPUs, confidential AI provides cryptographic guarantees that the cloud provider, the model serving infrastructure, and even the chip vendor cannot read the data being processed.

The technology stack is built on top of mature CPU TEE technologies (Intel TDX, AMD SEV-SNP) extended to GPUs through NVIDIA's Confidential Computing features on Hopper and Blackwell. The major cloud providers and several specialized vendors offer confidential AI services in production, and the most prominent consumer deployment (Apple's Private Cloud Compute) demonstrates that the approach can scale to mass-market use.

For practitioners, the relevant question is whether your use case requires confidential computing. For most users, the answer is no — the trust model of normal cloud LLM services is acceptable, and the overhead of confidential computing is unnecessary. For users in regulated industries, working with sensitive data, or with strong privacy requirements, the answer is increasingly yes. The tooling is good enough to deploy, the compliance story is improving, and the alternative — running inference entirely on-premises with smaller open-source models — comes with its own significant tradeoffs.

The broader significance is that confidential computing changes what is possible with cloud LLMs. The frontier models are too large and too expensive to run on customer hardware. Without confidential computing, the most sensitive workloads were excluded from these models entirely. With it, those workloads can use frontier models with verifiable confidentiality guarantees. This is a meaningful expansion of who can benefit from the best AI capabilities, and it is one of the more underappreciated trends in enterprise AI infrastructure today.
