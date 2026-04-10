# Federated Learning for LLMs: Training Across Organizational Boundaries

*April 2026 • Technical Report*

## 1. Introduction

Most LLM training assumes that all the data lives in one place. The pre-training corpus, the fine-tuning datasets, the human feedback signals, the synthetic generations — all of it is centralized in some operator's data warehouse, with the training cluster having direct access to read it. This assumption is so natural that it is rarely questioned. It is also wrong for a growing class of important use cases.

A hospital network may have valuable clinical data for fine-tuning a medical LLM, but cannot send patient records to a cloud provider. A consortium of banks might benefit from training on their combined transaction histories, but cannot share customer data with each other. A government agency may want to leverage data across multiple departments, but the legal restrictions on data sharing prevent any single entity from holding it all. In each case, the data is locked behind organizational and legal boundaries that the conventional centralized training pipeline cannot cross.

Federated learning is the technique developed to address this. Originally introduced by Google in 2017 for keyboard prediction on Android phones, federated learning trains a shared model across many separate data sources without ever centralizing the raw data. Each participant trains the model on its own data, computes the resulting weight updates, and shares only those updates with a coordinator. The coordinator aggregates the updates into a global model, sends the updated model back to the participants, and the process repeats. The raw data never leaves the participant's premises.

For LLMs, federated learning is harder than for the small models it was originally designed for, but the demand is pulling the techniques forward. This report examines what federated learning is, the specific challenges of applying it to LLMs, and the state of the practice in 2026.

## 2. The Federated Learning Concept

The basic federated learning loop is simple:

1. A coordinator sends the current model weights to all participants
2. Each participant trains the model on its local data for some number of steps
3. Each participant sends the resulting weight update (the difference between the new and old weights) back to the coordinator
4. The coordinator averages the updates (weighted by participant data size or other criteria) and applies them to produce a new model
5. Repeat until convergence

The privacy property comes from the fact that only model updates flow between participants and the coordinator — never the raw training data. Whether this constitutes meaningful privacy is a subtle question (covered later), but the basic property holds: a participant's data is processed only on their own infrastructure.

This pattern, called Federated Averaging or FedAvg, is the foundation of essentially all practical federated learning. Variations and refinements have been developed over the years, but the basic structure is consistent across implementations.

### 2.1 Cross-Device vs. Cross-Silo

Federated learning has two main flavors based on the participant population. **Cross-device federated learning** has many small participants, typically mobile phones or IoT devices. Each participant has limited compute, may drop in and out of training, and contributes a small amount of data. Google's original federated learning work for Gboard was cross-device, with millions of phones participating asynchronously.

**Cross-silo federated learning** has a few large participants, typically organizations with substantial compute and data. Each participant has reliable infrastructure, persistent participation, and large datasets. Healthcare and financial federated learning are typically cross-silo, with hospitals or banks as the participants.

For LLMs, cross-silo is the relevant flavor. Training a meaningful LLM requires substantial compute on each participant's side, which rules out the cross-device setting. The participants in a federated LLM project are organizations with their own GPU clusters, not individual users with phones.

## 3. The Challenges of Federated LLM Training

Federated learning was developed for small models — typically a few million parameters at most. LLMs are at least three orders of magnitude larger, and several aspects of the federated approach do not scale gracefully.

### 3.1 Communication Overhead

The fundamental cost of federated learning is the communication of model updates between participants and the coordinator. For a model with N parameters, each round of communication moves O(N) data per participant. For a 70B-parameter model in fp16, that is 140 GB per round per participant. With 10 participants and 100 rounds of training, the total communication is 280 TB — substantial but manageable over high-bandwidth links.

For frontier models in the 400B-parameter range, the communication cost becomes prohibitive. A federated training run of a 400B model with 10 participants and 1000 rounds of training would move 16 PB of data — at 100 Gbps, that is 14 days of continuous communication, on top of whatever compute the training itself requires.

The standard mitigation is to communicate only differences from the base model. If the base model is shared once at the start, subsequent rounds only need to send the parameter updates, which can be compressed (delta encoding, quantization, sparsification) much more aggressively than the full weights.

### 3.2 Heterogeneous Data

In conventional centralized training, the dataset is shuffled into batches and the model sees a representative sample of the entire distribution at every step. In federated learning, each participant only sees its own data, which may be drawn from a very different distribution than the other participants. The standard FedAvg algorithm averages updates from heterogeneous participants, which can lead to poor convergence when the data distributions differ significantly.

For LLMs, this is a serious issue. A medical LLM trained federally across hospitals will see very different patient populations at each site (urban academic medical center vs. rural community hospital, pediatric vs. geriatric, primary care vs. specialty care). The data heterogeneity can cause the global model to oscillate or fail to converge entirely.

Several techniques address this. FedProx adds a proximal term to each participant's local loss, penalizing weight updates that drift too far from the global model. SCAFFOLD uses control variates to correct for client drift. FedAvgM uses momentum at the global level. None of these fully solve the problem, and the best approach often depends on the specific data distribution.

### 3.3 Compute Heterogeneity

Different participants may have very different compute capabilities. A large hospital may have a 100-GPU cluster; a small clinic may have a single workstation. If the federated training requires all participants to do the same amount of work per round, the slowest participant becomes the bottleneck. Asynchronous federated learning, where participants update the global model independently rather than waiting for synchronization rounds, addresses this but introduces its own consistency challenges.

For LLM training, the compute requirements are usually high enough that all serious participants need substantial GPU clusters anyway. The heterogeneity problem is somewhat mitigated by self-selection — small participants do not join LLM federated training projects in the first place.

### 3.4 Privacy of Model Updates

The privacy guarantee of federated learning is that raw data is not shared. But model updates can leak information about the training data — sometimes a lot of information. Model inversion attacks have demonstrated that, given access to model gradients, an adversary can reconstruct training examples with surprising fidelity. This is a particular concern for LLMs, which can memorize training data verbatim.

Differential privacy provides a mathematical framework for limiting the information leaked by model updates. By adding carefully-calibrated noise to the gradients before sharing them, participants can bound the maximum information that any single update reveals. The catch is that differential privacy adds noise that hurts model quality, and the tradeoff between privacy and quality is unfavorable for very large models.

Secure aggregation protocols are another defense. These cryptographic protocols allow the coordinator to compute the sum (or weighted average) of participant updates without ever seeing the individual updates. Even if a participant's individual update leaks information, the coordinator only sees the aggregated sum, which leaks less. Secure aggregation is well-developed for moderate model sizes but has scalability challenges for billion-parameter models.

## 4. Federated Fine-Tuning

For LLMs, the most practical form of federated learning is federated fine-tuning rather than federated pre-training. Pre-training is too expensive to attempt federally for large models — the compute requirements alone make it impractical for most consortia. Fine-tuning is much cheaper and can be done effectively across organizational boundaries.

The typical federated fine-tuning workflow:

1. All participants start with the same pre-trained base model (downloaded from a public source or licensed from a model provider)
2. Each participant fine-tunes the model on its local data for a small number of steps
3. The fine-tuned weights (or LoRA adapters) are sent to the coordinator
4. The coordinator aggregates the updates and produces a new global model
5. Repeat for several rounds

Using LoRA adapters dramatically reduces the communication cost. Instead of sending the full model weights, participants send only the LoRA matrices, which are typically 100-1000× smaller than the full model. This makes federated fine-tuning of very large models practical even with modest network bandwidth between participants.

The Flower framework, OpenFL, NVIDIA FLARE, and several other open-source projects support federated LoRA fine-tuning of LLMs. The implementations are mature enough for production use in cross-silo healthcare and financial deployments.

## 5. Use Cases

### 5.1 Healthcare

The clearest use case for federated LLM training is healthcare. Hospitals have valuable training data (clinical notes, radiology reports, pathology findings) that cannot be shared due to HIPAA and similar regulations. Federated learning allows multiple hospitals to collaboratively train medical LLMs without any patient data leaving their systems.

Several real deployments exist. NVIDIA's FLARE framework has been used for federated radiology model training across major academic medical centers. The MELLODDY project (Machine Learning Ledger Orchestration for Drug Discovery) is a federated learning initiative across 10 pharmaceutical companies for drug discovery models. The Owkin platform provides federated ML services for healthcare consortia, with multiple production deployments in oncology and other specialties.

For LLMs specifically, federated medical fine-tuning is in its early stages. Several research projects have demonstrated federated fine-tuning of medical LLMs on clinical notes from multiple hospitals, with quality competitive with what would be achieved by pooling the data centrally. Production deployments are emerging.

### 5.2 Financial Services

Banks face similar constraints. Customer transaction data is heavily regulated (GDPR, CCPA, banking secrecy laws), and competitive pressures discourage banks from sharing data even when legally permissible. Federated learning offers a way for banks to collaboratively train fraud detection models, KYC systems, and customer service LLMs without exposing customer data to each other.

Several federated learning consortia have formed in the financial sector, though most are at the pilot stage. The use cases are primarily fraud detection and anti-money-laundering, where the value of cross-bank data is highest and the privacy concerns are most acute.

### 5.3 Government and Defense

Government agencies with classified data face the most extreme version of the federated learning problem. Different agencies may have data at different classification levels, and even within an agency, different programs may have data that cannot be combined. Federated learning is one of the few approaches that can leverage data across these boundaries without violating classification rules.

Specific deployments are not publicly disclosed, but several US government agencies have invested in federated learning research, and the technology is widely understood to be relevant to defense and intelligence applications.

## 6. The Privacy Question

How private is federated learning really? The answer is "more private than centralized training, but less private than the marketing suggests."

The privacy property of basic federated learning — that raw data does not leave the participant's premises — is real but limited. The model updates that do leave can leak meaningful information. Several attacks have been demonstrated:

- **Membership inference**: Determining whether a specific data point was used in training
- **Attribute inference**: Inferring properties of training data from the model's behavior
- **Model inversion**: Reconstructing training examples from the model
- **Property inference**: Determining global properties of the training distribution

For LLMs, these attacks are exacerbated by the model's tendency to memorize verbatim text. A federated LLM that has been trained on a hospital's clinical notes may, with the right prompts, regurgitate specific patient details. This is a real concern that goes beyond the basic privacy property of federated learning.

The defenses are differential privacy (which provably bounds the information leaked but hurts model quality), secure aggregation (which protects against the coordinator but not against participants), and various combinations of cryptographic techniques. None provide a complete solution, and the choice of which defenses to apply depends on the specific threat model.

For most practical deployments, the privacy property is best understood as "raw data is protected by organizational boundaries; model updates are protected by limited statistical leakage." This is meaningful protection compared to centralizing the data, but it is not equivalent to never having shared the data at all.

## 7. The Tooling Landscape

Several open-source frameworks support federated learning for LLMs:

**Flower**: A general-purpose federated learning framework with strong support for PyTorch and the LLM ecosystem. Includes federated LoRA examples and integration with Hugging Face.

**OpenFL (Open Federated Learning)**: An Intel-maintained framework focused on secure cross-silo federated learning for healthcare and other regulated industries.

**NVIDIA FLARE**: NVIDIA's federated learning framework, with emphasis on production deployment in healthcare and life sciences.

**FedML**: A federated learning platform with both open-source and commercial offerings, supporting LLMs and various other workloads.

**Substra**: A federated learning platform from the Owkin team, focused on healthcare research.

For most LLM use cases, Flower or NVIDIA FLARE are the practical starting points. Both have production-ready implementations of federated LoRA fine-tuning, integrate with the broader Hugging Face ecosystem, and support the privacy-enhancing techniques (differential privacy, secure aggregation) that are needed for sensitive deployments.

## 8. The Limits of Federation

Federated learning is not a universal solution. Several scenarios remain difficult or impractical:

**Pre-training**. Pre-training a foundation model from scratch in a federated setting is impractical for the largest models due to the communication overhead. Most federated LLM work assumes a publicly-available pre-trained base.

**Very heterogeneous data**. When participants' data distributions differ dramatically, the federated training can fail to converge or produce a global model that is worse than any individual participant's local model. Personalized federated learning (where each participant gets a customized version of the global model) addresses this but adds complexity.

**Adversarial participants**. If some participants are dishonest — sending corrupted updates to poison the global model, or trying to extract information from the updates of honest participants — the federated training can be compromised. Byzantine-robust aggregation methods exist but are not always sufficient.

**Continual learning**. Updating a federated model with new data over time is harder than the one-shot setting, because the model can forget earlier participants' contributions or oscillate between different distributions.

For these scenarios, alternatives like differential privacy on a centralized dataset, synthetic data generation, or homomorphic encryption may be more appropriate.

## 9. The Future

Federated learning for LLMs is at a similar stage to where centralized LLM training was around 2018: the techniques work for moderate-scale problems, the tooling is mature enough for production use in specific verticals, but the methods need refinement to scale to the largest models. Several developments are likely over the next several years:

- Better algorithms for handling heterogeneous data, reducing the convergence problems in cross-silo settings
- More efficient communication protocols, possibly using model-specific compression that exploits the structure of transformer weights
- Tighter integration with confidential computing (federated learning inside TEEs adds another layer of protection)
- Standardization of privacy-preserving aggregation protocols
- Production-grade implementations that are easy enough to use that they become a default choice for sensitive deployments

The biggest open question is whether federated learning can produce models that are competitive in quality with centrally-trained alternatives. For fine-tuning, the answer is increasingly yes. For pre-training, the answer is still no, and may remain so for the foreseeable future.

## 10. Conclusion

Federated learning for LLMs is a niche but important capability. It enables training on data that cannot be centralized, opening up valuable use cases in healthcare, finance, government, and other regulated industries. The techniques are well-understood, the tooling is production-ready for fine-tuning use cases, and the demand is growing as organizations look for ways to leverage AI without compromising data sovereignty.

For practitioners, the practical question is whether your use case has data that cannot be centralized but could benefit from being combined. If yes, federated learning is worth investigating. If no — if the data could be pooled centrally without legal or organizational obstacles — federated learning adds complexity and overhead without meaningful benefit. The technology is a tool for a specific problem, not a general improvement over centralized training.

The deeper significance is that federated learning represents a different architectural model for AI: one in which data ownership and model ownership can be separated, where multiple parties can collaboratively benefit from training without a single entity controlling all the data. As AI becomes more central to critical infrastructure, this distinction will matter more, and the tools that support distributed governance models will become more important.
