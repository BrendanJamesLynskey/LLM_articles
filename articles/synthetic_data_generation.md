# Synthetic Data Generation for LLM Training

*April 2026 · Technical Report*

## 1. Introduction

The development of large language models has been constrained by three persistent challenges: the availability of high-quality training data, the cost of human annotation, and the difficulty of generating diverse examples for specialized domains. As models have grown larger and training datasets have expanded to trillions of tokens, these constraints have become more acute — the supply of naturally occurring high-quality text on the internet is finite, human annotation is expensive and slow, and many important training domains (medical reasoning, mathematical proofs, code debugging) have relatively sparse coverage in web-scraped corpora.

Synthetic data generation — using AI models themselves to create training data — has emerged as a powerful and increasingly essential technique for addressing these constraints. From the early Self-Instruct method that generated instruction-following examples using GPT-3 to the sophisticated multi-stage pipelines that produced the Phi model family's "textbooks," synthetic data has become a core component of modern LLM training pipelines at every stage: pre-training, supervised fine-tuning, preference learning, and domain-specific specialization.

This report provides a comprehensive technical examination of synthetic data generation for LLM training: the methods for generating synthetic examples, the quality filtering techniques that make synthetic data useful, the applications across different training stages, the risks (model collapse, homogenization, legal concerns), and the tooling ecosystem that supports synthetic data workflows.

## 2. Why Synthetic Data

### 2.1 Data Scarcity in Specialized Domains

While the open internet provides billions of web pages suitable for general pre-training, many important capabilities require data that is scarce online. Advanced mathematical reasoning, formal proofs, complex multi-step coding problems with test cases, medical case analysis, legal reasoning, scientific methodology — these domains have limited high-quality natural text. Generating synthetic examples that target these specific capabilities allows training data to be purpose-built for the skills the model needs to learn.

The scarcity problem is compounded for instruction-following data. While the internet contains vast amounts of text, it contains relatively few examples of the kind of human-AI interaction patterns needed for instruction tuning: clear instructions paired with helpful, detailed, well-structured responses. Before synthetic data methods, creating instruction-tuning datasets required expensive human annotation campaigns (InstructGPT used approximately 13,000 human-written demonstrations and over 30,000 human comparisons).

### 2.2 Cost Reduction

Human annotation is expensive. Depending on the domain and quality requirements, annotating a single instruction-response pair costs $0.50 to $50, and preference annotations (comparing two responses and selecting the better one) cost $1 to $10 per comparison. At the scale needed for modern LLM training (hundreds of thousands to millions of examples), annotation budgets quickly reach hundreds of thousands or millions of dollars.

Synthetic data generation dramatically reduces these costs. Generating an instruction-response pair using a frontier API model costs $0.01 to $0.10, depending on length and model choice. Generating using open-weight models running on owned hardware reduces marginal costs further. The cost advantage is typically 10× to 1000×, depending on the domain and quality requirements.

### 2.3 Quality Control

Counterintuitively, synthetic data can sometimes be higher quality than human annotations for certain tasks. Human annotators vary in expertise, attention, and effort. They make errors, misunderstand instructions, and produce inconsistent outputs. A well-prompted frontier model can generate consistently structured, detailed, and accurate responses — particularly for tasks where the model's capabilities exceed those of a typical human annotator (such as writing well-documented code or providing comprehensive explanations of technical concepts).

This does not mean synthetic data is always better — human data provides diversity, creativity, and real-world groundedness that synthetic data may lack. But the quality control advantage is real for specific use cases and explains why synthetic data has become a standard component of training pipelines.

### 2.4 Controllable Diversity

Synthetic data generation allows precise control over the distribution of training examples. If a model is weak at multi-step reasoning, synthetic data can be generated specifically targeting that capability. If training data is underrepresented for a particular language, domain, or task format, synthetic examples can fill the gap. This controllability is particularly valuable for curriculum learning approaches that present training examples in a carefully sequenced order.

## 3. Self-Instruct and Alpaca

### 3.1 The Self-Instruct Method

Self-Instruct (Wang et al., December 2022) was the first widely adopted method for generating synthetic instruction-following data using language models. The method starts with a small seed set of 175 human-written tasks (instructions paired with input-output examples) and uses GPT-3 to generate new tasks iteratively:

1. **Task Generation**: The model is prompted with 8 randomly sampled seed tasks and asked to generate a new task (instruction + optional input + output).
2. **Classification**: A classifier determines whether the new task is a "classification" task (finite label set) or a "generation" task (open-ended output).
3. **Instance Generation**: For each new task, the model generates input-output instances.
4. **Filtering**: Duplicate, low-quality, and overly similar tasks are removed using ROUGE-L similarity and heuristic filters.

Starting from 175 seed tasks, Self-Instruct generated 52,000 instruction-following examples. When used to fine-tune GPT-3, the resulting model showed significant improvement in instruction-following ability, approaching (though not matching) the performance of InstructGPT, which used expensive human annotations.

### 3.2 Alpaca: Scaling Self-Instruct

Stanford's Alpaca (March 2023) applied the Self-Instruct methodology with a key improvement: using the much more capable text-davinci-003 (GPT-3.5) to generate the training data, then fine-tuning the smaller LLaMA-7B model on the result. Alpaca generated 52,000 instruction-following examples at a total cost of approximately $500, then fine-tuned LLaMA-7B on these examples for roughly $100 in compute.

The resulting Alpaca model demonstrated instruction-following capabilities qualitatively similar to GPT-3.5 for many common tasks, despite being a 7B parameter model trained on synthetic data generated by a much larger model. This was a watershed moment: it demonstrated that the knowledge and capabilities of a large model could be transferred to a smaller model through synthetic data, at dramatically lower cost than human annotation.

The Alpaca recipe became a template widely adopted by the open-source community. Dozens of models followed the pattern: use a frontier model to generate instruction-following data, then fine-tune an open-weight base model on that data. Vicuna, Dolly, Koala, and many other early instruction-tuned models used variations of this approach.

### 3.3 Limitations of Early Approaches

The Self-Instruct/Alpaca approach had important limitations:

- **Shallow diversity**: The generated instructions tended to cluster around common patterns, with limited coverage of complex, multi-step, or domain-specific tasks.
- **Quality ceiling**: The quality of generated examples was bounded by the capability of the generating model. Errors, hallucinations, and mediocre responses in the generated data were propagated to the student model.
- **Evaluator bias**: The generated data reflected the biases and preferences of the generating model, including its formatting preferences, verbosity tendencies, and knowledge gaps.
- **Legal uncertainty**: Using the output of proprietary models (GPT-3.5, GPT-4) as training data raised legal questions about the terms of service and intellectual property implications.

These limitations motivated the development of more sophisticated synthetic data generation methods.

## 4. Evol-Instruct and WizardLM

### 4.1 The Evol-Instruct Method

Evol-Instruct (Xu et al., April 2023), the technique behind WizardLM, addressed the diversity limitation of Self-Instruct by systematically evolving instructions to create more complex and varied training examples. Rather than generating instructions from scratch, Evol-Instruct starts with a pool of existing instructions and applies evolutionary operations to make them progressively more complex.

The method defines two types of evolution:

**In-Depth Evolution** makes instructions more complex through operations like:
- Adding constraints ("solve this math problem, but only using prime numbers")
- Deepening reasoning ("explain why, then provide a counterexample")
- Increasing complexity ("extend this to handle edge cases")
- Concretizing ("instead of a general sorting algorithm, implement one for a linked list")
- Adding multiple steps ("first do X, then analyze the result, then modify based on the analysis")

**In-Breadth Evolution** increases topic diversity by:
- Generating new instructions inspired by existing ones but covering different topics
- Varying the task format (from Q&A to analysis to creative writing)
- Changing the domain while maintaining similar complexity

### 4.2 The Evolution Process

The Evol-Instruct process is iterative:

1. Start with a seed set of instructions (Alpaca's 52K instructions were used in the original work).
2. For each instruction, randomly select an evolution operation (in-depth or in-breadth).
3. Prompt the generating model (ChatGPT/GPT-4) to apply the evolution, producing a new, more complex instruction.
4. Generate a response for the evolved instruction.
5. Filter out failed evolutions (instructions that are incoherent, impossible, or too similar to the original).
6. Repeat for multiple rounds, using the evolved instructions as the new seed set.

After several rounds of evolution, the resulting dataset contains instructions spanning a much wider range of complexity levels — from simple factual questions to complex multi-step reasoning tasks — than the original seed set.

### 4.3 WizardLM Results

WizardLM, fine-tuned on 250K Evol-Instruct examples, demonstrated significantly stronger instruction-following capabilities than Alpaca and other Self-Instruct-based models, particularly on complex tasks requiring multi-step reasoning. The improvement was most pronounced for difficult instructions — the evolutionary process had successfully created training examples that taught the model to handle complexity.

The WizardLM approach was extended to code (WizardCoder, using code-specific evolution operations like adding algorithmic constraints or increasing time/space complexity requirements) and mathematics (WizardMath, using mathematical evolution operations like increasing proof rigor or adding mathematical constraints). Both demonstrated that domain-specific evolution strategies could produce synthetic training data that significantly improved model performance in specific domains.

## 5. Distillation from Stronger Models

### 5.1 Knowledge Distillation via Synthetic Data

The most common form of synthetic data generation is knowledge distillation — using a stronger model (the "teacher") to generate training data for a weaker model (the "student"). This is distinct from classical knowledge distillation (which transfers knowledge through soft probability distributions over the vocabulary) in that the transfer happens through generated text: the teacher produces examples that the student learns to imitate.

The basic recipe is:

1. **Curate prompts**: Assemble a diverse set of instructions or prompts covering the desired capabilities.
2. **Generate responses**: Feed each prompt to the teacher model and collect the generated responses.
3. **Filter and curate**: Remove low-quality, incorrect, or problematic responses.
4. **Train the student**: Fine-tune the student model on the curated prompt-response pairs.

This approach has been used at massive scale. The Orca models (Microsoft) used millions of synthetic examples generated by GPT-4, with carefully designed system prompts that encouraged the teacher to produce detailed explanations of its reasoning process. The Phi models used synthetic "textbooks" and exercises generated by GPT-4. Numerous open-source models have been fine-tuned on synthetic data generated by Claude, GPT-4, or other frontier models.

### 5.2 Prompt Engineering for Teacher Models

The quality of synthetic training data depends heavily on how the teacher model is prompted. Key strategies include:

**System prompts that shape response style**: Instructing the teacher to "think step by step," "explain your reasoning in detail," or "provide a comprehensive answer with examples" produces more informative training examples than bare responses.

**Role-based prompting**: Asking the teacher to respond "as an expert in X" or "as if explaining to a graduate student" shapes the level of detail and expertise in the responses.

**Format specifications**: Requiring specific output formats (structured JSON, markdown with headers, code with comments) ensures the student learns to produce well-structured outputs.

**Chain-of-thought prompting**: Asking the teacher to show its reasoning process produces training examples that teach the student not just what to answer but how to reason.

### 5.3 The Orca Approach

Microsoft's Orca (June 2023) introduced a systematic approach to distillation that emphasized learning the reasoning process, not just the final answer. Orca used carefully crafted system prompts to elicit detailed explanations from GPT-4:

- "You are a helpful assistant. Think step by step and explain your reasoning clearly."
- "Explain the problem-solving process, including any intermediate steps or considerations."

The resulting training data contained not just correct answers but detailed explanations of the reasoning behind those answers. When used to fine-tune a 13B parameter model, this approach produced a student that significantly outperformed models trained on standard instruction-following data, particularly on reasoning-heavy tasks.

Orca 2 extended this by teaching the student model to use different reasoning strategies for different types of problems — step-by-step reasoning for math, extractive reasoning for factual questions, creative generation for open-ended tasks. The system prompts for the teacher were varied to produce diverse reasoning strategies in the training data.

### 5.4 Multi-Teacher Distillation

Some approaches use multiple teacher models to generate synthetic data, combining the strengths of different models. The Airoboros dataset, for example, combined outputs from multiple frontier models, selecting the best response for each prompt. This multi-teacher approach can reduce the biases and weaknesses of any single teacher model.

## 6. Constitutional AI Data Generation

### 6.1 The Constitutional AI Framework

Anthropic's Constitutional AI (CAI) framework uses synthetic data generation as a core component of alignment training. Rather than relying entirely on human feedback to teach models to be helpful, harmless, and honest, CAI uses the model itself to generate and evaluate training data according to a set of principles (the "constitution").

The CAI synthetic data pipeline has two phases:

**Supervised Learning Phase (SL-CAI)**:
1. Start with a helpful but potentially harmful model.
2. Generate responses to red-teaming prompts (prompts designed to elicit harmful outputs).
3. Ask the model to critique its own response according to constitutional principles ("Identify specific ways in which the assistant's response is harmful, unethical, or problematic").
4. Ask the model to revise its response based on the critique.
5. Fine-tune the model on the revised (improved) responses.

**Reinforcement Learning Phase (RL-CAI)**:
1. Generate pairs of responses to the same prompt.
2. Ask the model to evaluate which response better adheres to the constitutional principles.
3. Use these synthetic preference labels to train a reward model.
4. Use the reward model for RLHF training.

### 6.2 Self-Critique and Revision

The self-critique mechanism is powerful because it allows the model's own understanding of ethics and safety to generate training signal. The model identifies problems in its responses, explains why they are problematic (citing specific constitutional principles), and produces improved versions. This creates a synthetic dataset of (harmful response, improved response) pairs that teaches the model to self-correct.

The quality of self-critique depends on the model's capability. More capable models produce more nuanced critiques and better revisions, creating a virtuous cycle where improvement in capability leads to improvement in alignment training data. This has been one of the drivers of the iterative training approach used by Anthropic, where each generation of Claude produces better training data for the next.

### 6.3 RLAIF: AI Feedback as Synthetic Preference Data

Reinforcement Learning from AI Feedback (RLAIF) extends the CAI principle to preference data. Instead of having humans compare model responses and select the better one, an AI model makes the comparison. This dramatically reduces the cost of preference data generation (from dollars per comparison to fractions of a cent) and enables much larger preference datasets.

RLAIF has been shown to produce reward models and aligned models that are competitive with — and in some cases indistinguishable from — those trained on human feedback, particularly for straightforward helpfulness and safety judgments. Human feedback remains valuable for subtle or contested cases where human judgment provides signal that models cannot reliably generate.

## 7. Quality Filtering

### 7.1 The Importance of Filtering

Synthetic data, like all data, follows the principle of "garbage in, garbage out." Raw synthetic data generated by even the best teacher models contains errors, hallucinations, formatting issues, low-quality responses, and examples that could teach undesirable behaviors. Quality filtering is therefore as important as data generation — and in many cases more important, since a smaller, high-quality dataset often produces better results than a larger, noisier one.

### 7.2 Reward Model Scoring

One of the most effective filtering approaches uses a trained reward model to score synthetic examples. Each generated response is scored by the reward model, and only examples above a quality threshold are retained. This approach leverages the reward model's training on human preference data to identify high-quality responses.

The reward model can be used in several ways:
- **Hard threshold**: Retain only examples with reward scores above a fixed threshold.
- **Top-K selection**: For each prompt, generate multiple responses and retain only the highest-scored one.
- **Rejection sampling**: Generate responses until one exceeds the quality threshold, discarding rejected samples.

Rejection sampling with a reward model is particularly powerful. By generating multiple responses per prompt and selecting the best, the effective quality of the synthetic data can significantly exceed the average quality of the generating model. This is sometimes called "best-of-N" sampling and is a key component of many synthetic data pipelines.

### 7.3 LLM-as-Judge

Using a language model to evaluate the quality of synthetic examples has become a standard filtering technique. The judge model (typically a frontier model) evaluates each example on dimensions like helpfulness, accuracy, clarity, and completeness, providing a quality score and optionally an explanation of the assessment.

Common evaluation criteria include:
- **Instruction following**: Does the response address all parts of the instruction?
- **Factual accuracy**: Is the information correct? (Particularly important for knowledge-heavy domains.)
- **Reasoning quality**: Are logical steps valid and clearly explained?
- **Completeness**: Does the response cover the topic adequately?
- **Formatting**: Is the response well-structured and easy to read?

### 7.4 Deduplication

Synthetic data pipelines can produce many near-duplicate examples, particularly when the generating model has strong default patterns. Deduplication is essential to prevent the student model from overfitting to repeated patterns. Techniques include:

- **Exact deduplication**: Remove identical examples.
- **Near-duplicate detection**: Use MinHash, SimHash, or embedding-based similarity to identify and remove near-duplicates.
- **Semantic deduplication**: Use embedding models to cluster semantically similar examples and retain only representative examples from each cluster.
- **N-gram overlap filtering**: Remove examples with high n-gram overlap with existing examples in the dataset.

### 7.5 Difficulty Balancing

A well-constructed synthetic dataset should span a range of difficulty levels. Filtering can be used to ensure this balance — removing very easy examples (which add little training signal) and optionally removing examples that are too difficult (which may contain errors or be beyond the student model's learning capacity). Curriculum learning approaches sequence examples from easy to hard, but this requires first classifying examples by difficulty.

### 7.6 Contamination Checking

Synthetic data generated by models trained on benchmark datasets may contain examples that are too similar to benchmark test examples, creating artificial inflation of benchmark scores. Contamination checking — comparing generated examples against test sets and removing those that are too similar — is an important but often neglected filtering step.

## 8. Synthetic Preference Data for RLHF and DPO

### 8.1 Generating Preference Pairs

Preference data — pairs of responses where one is labeled as better than the other — is essential for RLHF and DPO training. Synthetic preference data can be generated in several ways:

**Contrastive generation**: Generate a good response (using careful prompting) and a bad response (using prompting that encourages common errors) for the same prompt. The good/bad labels come from the generation process.

**Best-of-N with ranking**: Generate N responses to each prompt, score them with a reward model, and create preference pairs from the highest-scoring and lowest-scoring responses.

**Model-as-judge comparison**: Generate two responses and ask a judge model to determine which is better, creating a preference label.

**Perturbation-based**: Start with a good response and systematically degrade it (introduce errors, remove details, add hallucinations) to create a worse version, forming a preference pair.

### 8.2 DPO-Specific Data Generation

Direct Preference Optimization (DPO) requires pairs of (chosen, rejected) responses for each prompt. The quality of DPO training depends heavily on the quality and informativeness of these pairs. Effective DPO data has several properties:

- **Both responses should be plausible**: If the rejected response is obviously terrible, the model learns very little from the pair.
- **The difference should be meaningful**: The chosen response should be better in a specific, learnable way (more accurate, better reasoned, safer, more helpful).
- **Diverse failure modes**: The rejected responses should exhibit diverse problems, not just one type of error.

### 8.3 Iterative Preference Data

A powerful approach generates preference data iteratively. After initial training, the model's own outputs become the candidates for preference evaluation:

1. Train an initial model.
2. Sample responses from the model.
3. Evaluate responses (using humans, reward models, or judge models).
4. Create preference pairs from the model's own good and bad responses.
5. Train the model further using DPO or RLHF on these pairs.
6. Repeat.

This iterative self-improvement loop, sometimes called "online DPO" or "iterative DPO," generates preference data that is specifically targeted at the model's current weaknesses — the preference pairs are drawn from the model's own output distribution, making them maximally informative for learning.

### 8.4 UltraFeedback and Community Datasets

UltraFeedback (Cui et al., 2023) is a prominent example of a large-scale synthetic preference dataset. It contains approximately 64K prompts, each with 4 responses from different models, scored by GPT-4 on dimensions of instruction-following, truthfulness, honesty, and helpfulness. The scores are converted to preference pairs for DPO training. Zephyr-7B, one of the first strong 7B instruction-following models, was trained using UltraFeedback-derived preference data, demonstrating the effectiveness of synthetic preference data for alignment.

## 9. Code Data Synthesis

### 9.1 Why Synthetic Code Data

Code is a particularly promising domain for synthetic data generation because:

- **Verifiability**: Generated code can be automatically tested by running it against test cases, providing an objective quality signal.
- **Structured output**: Code follows strict syntax and semantics, making quality assessment more tractable.
- **Scarcity of complex examples**: While GitHub contains enormous amounts of code, well-documented solutions to complex algorithmic problems with step-by-step explanations are relatively scarce.
- **Diversity of approaches**: Multiple correct solutions exist for most coding problems, enabling diverse synthetic data.

### 9.2 Code Generation Strategies

**Problem-Solution Generation**: Generate coding problems at various difficulty levels, then generate solutions. Problems can be generated from scratch ("Generate a medium-difficulty dynamic programming problem") or by evolving existing problems (adding constraints, changing data structures, combining multiple problems).

**Test-Case-First Generation**: Generate test cases first, then generate code that passes them. This ensures the generated code is correct (it's verified against the tests) and provides both training examples and evaluation infrastructure.

**Code Explanation Generation**: Given existing code (from open-source repositories), generate detailed explanations, comments, and step-by-step walkthroughs. This produces training data that teaches the model to explain code, not just generate it.

**Bug Introduction and Fixing**: Start with correct code, systematically introduce bugs, then pair the buggy code with the fix. This creates training data for code debugging — a critical capability that is underrepresented in naturally occurring data.

**Code Translation**: Translate code between programming languages, generating parallel examples that teach cross-language understanding.

### 9.3 OSS-Instruct

OSS-Instruct (Wei et al., 2024), the technique behind Magicoder, addresses the quality problem by grounding synthetic code generation in real open-source software. Rather than generating coding problems from scratch (which tends to produce generic, textbook-style problems), OSS-Instruct samples real code snippets from open-source repositories and uses them as inspiration for generating more diverse and realistic coding problems.

The process:
1. Sample a code snippet from open-source repositories.
2. Prompt the teacher model: "Inspired by this code snippet, generate a programming problem and its solution."
3. The teacher generates a problem that relates to the concepts in the snippet but is a new, self-contained problem.
4. Filter and validate the generated problem-solution pairs.

This produces coding problems that reflect real-world software patterns rather than artificial textbook exercises, resulting in more practical and diverse training data.

### 9.4 Execution-Based Filtering

A key advantage of code synthetic data is the ability to automatically verify correctness by executing generated code against test cases. Pipelines that incorporate execution-based filtering can:

- Run generated code and check for syntax errors, runtime errors, and incorrect output.
- Verify that generated solutions pass automatically generated or hand-crafted test cases.
- Check for common issues like infinite loops (using timeouts), memory leaks, and security vulnerabilities.
- Validate that generated code handles edge cases.

This execution-based filtering produces significantly higher-quality code training data than text-based quality filtering alone.

## 10. Math Data Generation

### 10.1 The Math Data Challenge

Mathematical reasoning is one of the most challenging capabilities for LLMs, and high-quality math training data is scarce. While the internet contains math problems, the solutions are often incomplete (showing final answers without intermediate steps), incorrect, or at difficulty levels that don't cover the full range of mathematical reasoning.

### 10.2 Synthetic Math Problem Generation

Approaches to generating synthetic math data include:

**Template-based generation**: Define mathematical problem templates with variable parameters and generate instances by sampling parameter values. This ensures correctness (the template includes the solution method) but produces limited diversity.

**Evolutionary problem generation**: Start with existing problems and systematically modify them — changing numbers, adding constraints, combining problems, or increasing difficulty. This produces diverse problems while maintaining mathematical validity.

**Backward generation**: Start with a desired answer or mathematical concept and generate a problem that requires that concept to solve. This ensures coverage of specific mathematical skills.

**Multi-step problem construction**: Build complex problems by composing simpler sub-problems, creating training examples that require extended chains of mathematical reasoning.

### 10.3 Verification of Mathematical Solutions

Mathematical solutions can be partially verified automatically:

- **Numerical verification**: Check that the numerical answer is correct (for problems with deterministic answers).
- **Symbolic verification**: Use computer algebra systems (SymPy, Mathematica) to verify algebraic manipulations and equations.
- **Proof verification**: For formal proofs, use proof assistants (Lean, Coq) to verify correctness.
- **Step-by-step checking**: Use a judge model to verify each step of a solution, flagging steps that appear incorrect.

The combination of generated solutions with automated verification produces higher-quality math training data than generation alone.

### 10.4 MetaMath and Augmentation

MetaMath (Yu et al., 2023) demonstrated that augmenting existing math problems with rephrasings and answer-augmented versions could significantly improve mathematical reasoning. Given a math problem, MetaMath generates multiple versions: rephrased problems (same math, different wording), backward problems (given the answer, reconstruct the problem setup), and self-verification problems (verify whether a given answer is correct). This augmentation strategy, while simpler than full problem generation, proved remarkably effective — MetaMath-trained models achieved significant improvements on GSM8K and MATH benchmarks.

## 11. Textbooks Are All You Need: The Phi Model Approach

### 11.1 The Phi Hypothesis

Microsoft's Phi model series (Phi-1, Phi-1.5, Phi-2, Phi-3, Phi-4) is built on a striking hypothesis: a small model trained on high-quality synthetic data can match or exceed much larger models trained on web-scraped data. The "textbooks" in "Textbooks Are All You Need" refers to the quality of the synthetic training data — structured, pedagogical, comprehensive explanations of concepts, as opposed to the noisy, incomplete, and often incorrect information found on the internet.

### 11.2 Phi-1: Code Textbooks

Phi-1 (June 2023), a 1.3B parameter model, demonstrated surprisingly strong coding performance by training primarily on synthetic data. The training data consisted of:

- **"Textbook" data**: Approximately 1B tokens of synthetic Python textbook content generated by GPT-3.5, covering programming concepts from basic to advanced with clear explanations and examples.
- **"Exercise" data**: Approximately 180M tokens of synthetic Python exercises with solutions, generated to practice the concepts covered in the textbooks.
- **Filtered web data**: Approximately 6B tokens of code data from The Stack, filtered for quality using a classifier trained to distinguish high-quality code from low-quality code.

Despite being orders of magnitude smaller than models like StarCoder (15B) and Code Llama (34B), Phi-1 achieved competitive or superior performance on HumanEval and MBPP code benchmarks.

### 11.3 Scaling the Approach: Phi-2 through Phi-4

**Phi-2** (December 2023, 2.7B parameters) extended the approach to general language understanding. The synthetic training data included textbook-quality explanations across diverse subjects — science, mathematics, history, social studies — augmented with carefully filtered web data. Phi-2 matched or exceeded the performance of models 10× its size (including Llama 2-13B and Mistral-7B) on many benchmarks.

**Phi-3** (April 2024, available in 3.8B and 14B variants) further refined the approach with improved synthetic data generation. Phi-3 used a more sophisticated pipeline that included:
- Multi-step synthetic data generation with verification
- Data quality scoring using multiple judge models
- Curriculum-based data ordering (presenting simpler concepts before complex ones)
- Domain-balanced data mixing

**Phi-4** (December 2024, 14B parameters) pushed the approach further, achieving performance competitive with models several times its size on mathematical reasoning, coding, and scientific benchmarks. Phi-4's training included a particularly large proportion of synthetic data — reportedly over 40% of its training tokens were synthetically generated, with heavy emphasis on mathematical and reasoning-focused content.

### 11.4 Implications

The Phi model series demonstrated that data quality can substitute for model scale to a significant extent. A small model trained on carefully curated, high-quality synthetic data can match the performance of a much larger model trained on noisy web data. This has profound implications for the economics of LLM training and deployment: if equivalent capability can be achieved with smaller models, inference costs decrease proportionally, enabling deployment in resource-constrained environments.

However, the Phi approach has limitations. The models' benchmark performance sometimes exceeds their practical utility — they may perform well on structured benchmark tasks while struggling with the messiness and diversity of real-world usage. And the reliance on frontier models (GPT-3.5/GPT-4) for synthetic data generation creates a dependency on proprietary systems.

## 12. Risks and Challenges

### 12.1 Model Collapse

Model collapse is the phenomenon where a model trained on synthetic data from a previous-generation model progressively loses diversity and degrades in quality over successive generations. If Model A generates training data for Model B, and Model B generates training data for Model C, each generation may amplify the biases and limitations of the previous generation while losing the diversity of the original training distribution.

Shumailov et al. (2024) formalized this concern, showing that iterative training on model-generated data leads to progressive narrowing of the output distribution — the model "forgets" the tails of the original distribution and converges to a less diverse, lower-quality mode. This manifests as:

- Reduced vocabulary diversity (using fewer unique words and phrases)
- Increased repetition of common patterns
- Loss of minority perspectives and edge cases
- Degradation of factual accuracy on less common topics

### 12.2 Mitigating Model Collapse

Several strategies mitigate model collapse:

- **Mixing real and synthetic data**: Including a substantial proportion of real (non-synthetic) data in every training run prevents complete loss of the original data distribution.
- **Diversity constraints**: Explicitly measuring and maintaining diversity in synthetic data (unique n-grams, topic distribution, response format variety).
- **Fresh generation**: Generating synthetic data from the strongest available model rather than iteratively from each generation.
- **Quality over quantity**: Using smaller amounts of high-quality synthetic data rather than large amounts of degraded synthetic data.
- **Watermarking and provenance tracking**: Identifying and filtering out content that was itself generated by models, preventing recursive contamination.

### 12.3 Homogenization

Even without model collapse, synthetic data can homogenize model behavior. Models trained primarily on synthetic data may converge to a similar "voice" — the style and patterns of the teacher model. This is visible in the open-source model ecosystem, where many models fine-tuned on GPT-4-generated data produce similar-sounding responses with similar structural patterns (numbered lists, "Let me break this down," etc.).

Homogenization reduces the diversity of the AI ecosystem. If all models are trained on data from a small number of teacher models, the resulting models may share the same biases, blind spots, and failure modes. Strategies to mitigate homogenization include using diverse teacher models, incorporating human-written data, and explicitly encouraging stylistic diversity in synthetic data generation.

### 12.4 Hallucination Propagation

Teacher models hallucinate, and these hallucinations become "facts" in the synthetic training data. If GPT-4 generates a training example that contains a factual error, and a student model is trained on that example, the student learns the error as if it were true. This is particularly problematic for domains where hallucinations are difficult to detect (obscure historical facts, nuanced scientific claims, edge cases in law).

Verification and fact-checking of synthetic data mitigate this risk, but they add cost and complexity to the pipeline and cannot catch all errors. For safety-critical applications, over-reliance on synthetic data is a recognized risk.

### 12.5 Reward Hacking in Synthetic Preference Data

When synthetic preference data is generated using reward models or judge models as evaluators, the training process can optimize for the evaluator's preferences rather than genuine quality. The model learns to produce outputs that score well according to the evaluator's criteria, which may not align perfectly with actual user preferences. This is a form of Goodhart's law: when a measure becomes a target, it ceases to be a good measure.

Specific manifestations include:
- Excessive verbosity (longer responses often score higher with reward models)
- Sycophantic behavior (agreement and flattery score higher than honest disagreement)
- Format gaming (producing outputs with the structural patterns the reward model favors)

## 13. Tooling and Infrastructure

### 13.1 distilabel

distilabel (developed by Argilla) is an open-source framework specifically designed for synthetic data generation and quality annotation. It provides:

- **Pipeline abstractions**: Define multi-step data generation workflows (generate → evaluate → filter → augment) as composable pipelines.
- **Model integrations**: Built-in support for calling various LLM APIs (OpenAI, Anthropic, Cohere) and local models (vLLM, TGI) as generators or evaluators.
- **Task templates**: Pre-built templates for common generation tasks (instruction generation, preference pair creation, quality scoring).
- **Structured outputs**: Support for generating structured data (JSON, typed fields) for consistent output formatting.
- **Scalability**: Parallel execution across multiple model instances for high-throughput generation.

### 13.2 Argilla

Argilla is an open-source data labeling and curation platform that integrates with synthetic data workflows. It provides:

- **Annotation interfaces**: Web-based interfaces for human review and annotation of synthetic data.
- **Quality review workflows**: Tools for human reviewers to validate, correct, or reject synthetic examples.
- **Integration with datasets**: Direct integration with Hugging Face Datasets for easy data publishing and sharing.
- **Feedback collection**: Structured feedback collection that can be used to improve synthetic data generation.

### 13.3 Synthetic Data Libraries

Several other tools support synthetic data generation:

- **Genstruct**: Generates instruction-following data from raw text by identifying implicit questions and answers.
- **CAMEL**: A multi-agent framework for synthetic data generation, using role-playing conversations between AI agents.
- **Self-Play**: Frameworks where a model generates both sides of a conversation or debate, producing training data from self-interaction.
- **Magpie**: A technique for extracting synthetic instruction data by prompting models with only their system prompt and collecting the self-generated instructions.

### 13.4 Scaling Infrastructure

Large-scale synthetic data generation requires significant infrastructure:

- **API management**: Handling rate limits, retries, and cost optimization when using API-based teacher models.
- **Local model serving**: Running open-weight teacher models on GPU clusters using vLLM, TGI, or SGLang for high-throughput generation.
- **Storage and versioning**: Managing large synthetic datasets with versioning, lineage tracking, and quality metadata.
- **Pipeline orchestration**: Coordinating multi-step generation, evaluation, and filtering workflows.

## 14. Regulatory Considerations

### 14.1 Terms of Service

Using the output of commercial AI models as training data raises questions about compliance with terms of service. OpenAI's usage policies, for example, have provisions regarding the use of model outputs to train competing models. While enforcement is limited and the legal landscape is evolving, organizations generating synthetic data from proprietary models should review the applicable terms of service.

### 14.2 Intellectual Property

The intellectual property status of synthetic data is legally unsettled. Questions include:
- Does synthetic text generated by an AI model receive copyright protection?
- Does training on synthetic data generated by a copyrighted model create a derivative work?
- If synthetic data is based on copyrighted source material (e.g., a model generates training data inspired by a copyrighted textbook), does this constitute infringement?

These questions are being adjudicated in courts and regulatory bodies worldwide, with no clear consensus as of early 2026.

### 14.3 Transparency and Disclosure

Regulatory frameworks, including the EU AI Act, are moving toward requirements for transparency about the use of synthetic data in AI training. Organizations may need to disclose what proportion of training data is synthetic, what models were used to generate it, and what quality control measures were applied. Maintaining detailed provenance records for synthetic data is therefore prudent both for compliance and for internal quality management.

### 14.4 Model Output Disclosure

Some jurisdictions require or encourage disclosure when AI-generated content is used in downstream applications. If a model trained on synthetic data is deployed in a regulated context (healthcare, finance, legal), the synthetic nature of the training data may need to be disclosed as part of model documentation and risk assessment.

## 15. Best Practices

### 15.1 Data Pipeline Design

Effective synthetic data pipelines follow a structured workflow:

1. **Define objectives**: Identify the specific capabilities the synthetic data should teach.
2. **Curate seed data**: Assemble a diverse, representative seed set of prompts/instructions.
3. **Generate diverse examples**: Use techniques like Evol-Instruct to ensure diversity in difficulty, format, and topic.
4. **Multi-model generation**: Use multiple teacher models to reduce single-model biases.
5. **Quality filtering**: Apply multiple quality signals (reward model scores, judge model evaluations, execution tests for code, factual verification).
6. **Deduplication**: Remove near-duplicates and enforce diversity.
7. **Human review**: Sample and manually review a subset of the generated data.
8. **Mix with real data**: Combine synthetic data with high-quality human-generated data.
9. **Iterate**: Use the trained model's weaknesses to guide the next round of synthetic data generation.

### 15.2 Quality Metrics

Track metrics throughout the pipeline:
- **Generation statistics**: Success rate, average length, diversity measures.
- **Quality scores**: Distribution of reward model or judge model scores.
- **Diversity measures**: Unique n-grams, topic distribution, format variety.
- **Downstream impact**: A/B testing of models trained with and without the synthetic data on evaluation benchmarks.

### 15.3 Documentation and Reproducibility

Document all aspects of the synthetic data pipeline:
- Models used for generation and their versions/configurations.
- Prompts and system prompts used.
- Filtering criteria and thresholds.
- Mixing ratios with other data sources.
- Known limitations and failure modes.

## 16. Conclusion

Synthetic data generation has evolved from a niche technique to a fundamental component of modern LLM training. The progression from Self-Instruct's simple recipe to the sophisticated multi-stage pipelines behind models like Phi-4 reflects a maturing understanding of how to generate, filter, and use synthetic data effectively.

The key insight underlying all synthetic data methods is that the knowledge and capabilities of a strong model can be distilled into training data — and that this training data can be more effective than equivalent amounts of naturally occurring text, because it can be targeted, curated, and quality-controlled in ways that web-scraped data cannot.

At the same time, synthetic data carries real risks. Model collapse, homogenization, hallucination propagation, and reward hacking are not theoretical concerns but practical challenges that require deliberate mitigation. The dependence on frontier proprietary models for generation raises legal and strategic questions. And the "textbook quality" hypothesis, while validated by the Phi models, has limitations — synthetic data excels at teaching structured knowledge and reasoning patterns but may not capture the full diversity and messiness of human communication.

The field is moving toward more sophisticated pipelines that combine synthetic and natural data, use multiple teacher models and diverse generation strategies, apply multi-stage quality filtering, and iterate based on the trained model's observed weaknesses. As these pipelines mature, synthetic data will likely constitute an increasing share of LLM training data — not as a replacement for natural data, but as a carefully engineered complement that addresses specific capability gaps and quality requirements.

## References

1. Wang, Y., et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." ACL 2023.
2. Taori, R., et al. "Stanford Alpaca: An Instruction-following LLaMA Model." 2023.
3. Xu, C., et al. "WizardLM: Empowering Large Language Models to Follow Complex Instructions." 2023.
4. Mukherjee, S., et al. "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." 2023.
5. Bai, Y., et al. "Constitutional AI: Harmlessness from AI Feedback." 2022.
6. Lee, H., et al. "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." 2023.
7. Gunasekar, S., et al. "Textbooks Are All You Need." 2023.
8. Li, Y., et al. "Textbooks Are All You Need II: phi-1.5 Technical Report." 2023.
9. Javaheripi, M., et al. "Phi-2: The Surprising Power of Small Language Models." 2023.
10. Abdin, M., et al. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." 2024.
11. Abdin, M., et al. "Phi-4 Technical Report." 2024.
12. Shumailov, I., et al. "The Curse of Recursion: Training on Generated Data Makes Models Forget." 2024.
13. Cui, G., et al. "UltraFeedback: Boosting Language Models with High-quality Feedback." 2023.
14. Wei, Y., et al. "Magicoder: Source Code Is All You Need." 2024.
15. Yu, L., et al. "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models." 2023.
