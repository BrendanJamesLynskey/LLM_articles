# Photonic AI Accelerators: Computing with Light

*April 2026 • Technical Report*

## 1. Introduction

For sixty years, computing has been an electrical phenomenon. Electrons move through silicon transistors, voltages encode bits, and the physics of doped semiconductors determines what is possible. Every commercial AI accelerator built today — NVIDIA GPUs, Google TPUs, Cerebras WSEs, Groq LPUs — is built on this foundation. They differ in architecture, packaging, and optimization targets, but at the lowest level they all do the same thing: move electrons through transistors, very fast.

A small but well-funded group of companies is building something fundamentally different. Photonic AI accelerators perform computation with light. Instead of routing voltages through wires, they route optical signals through waveguides. Instead of multiplying numbers in transistors, they multiply them through interference patterns of laser beams. The physics is alien to anyone trained on conventional digital design, but the potential rewards — orders of magnitude lower energy per operation, near-zero latency on data movement, and the ability to perform certain operations (matrix multiplication, in particular) in essentially constant time — have attracted serious investment from venture capital, hyperscalers, and government research funding.

Photonic AI is not yet a commercial reality at scale. The first products are shipping in limited quantities to early customers, and most of the capability is still proprietary research at companies like Lightmatter, Lightelligence, Ayar Labs, Q.ANT, and PsiQuantum. But the trajectory is real, and if the engineering challenges can be overcome, photonic accelerators may carve out an important niche in AI inference over the next decade.

This report examines what photonic AI accelerators are, how they differ from electronic ones, what they are good (and bad) at, and where the technology stands in 2026.

## 2. Why Photons

The case for computing with light rests on several physical advantages.

### 2.1 No Resistive Heating

Electrical signals in metal wires lose energy to resistance. A modern GPU spends a substantial fraction of its power budget — perhaps 30-50% — moving data between memory and compute units, with most of that energy dissipated as heat in the interconnect. Optical signals in waveguides do not suffer from resistive losses in the same way. A photon traveling through a glass or silicon waveguide loses very little energy, and the energy budget for moving optical data is dominated by the cost of generating the light in the first place (at the laser source) and detecting it (at the photodetector).

For AI workloads, where data movement dominates energy consumption, the potential savings are enormous. A photonic interconnect can deliver data at 1 pJ/bit or less, compared to 5-10 pJ/bit for the best electrical alternatives. This is a 5-10× improvement in energy efficiency for data movement alone.

### 2.2 Native Parallelism

Light has an unusual property: multiple wavelengths can travel through the same waveguide simultaneously without interfering with each other. This is wavelength-division multiplexing (WDM), and it underpins all of fiber-optic telecommunications. For computing, it means that a single photonic data path can carry many independent data streams in parallel — potentially 32, 64, or more wavelengths per waveguide.

For matrix multiplication in particular, WDM enables a striking trick. The multiply-accumulate operations of a matrix product can be performed by encoding values onto different wavelengths, sending them through interferometric devices that combine their amplitudes, and reading the result. The computation happens at the speed of light through the interferometer — typically tens of picoseconds — independent of the matrix size.

### 2.3 Speed of Data Movement

Light in a silicon waveguide travels at roughly 80,000 km/s (about a quarter of the speed of light in vacuum due to the higher refractive index). This is fast, but more importantly, it is independent of the distance. An optical signal travels across a 10mm chip in 125 picoseconds and across a 1m fiber in 12.5 nanoseconds. There is no equivalent of capacitive charging delay or skin-effect losses that grow with distance. For chip-to-chip communication, this means that physical distance is much less important to performance than it is for electrical signals.

## 3. The Physics of Photonic Matrix Multiplication

The most natural operation for a photonic accelerator is matrix-vector multiplication. The technique was demonstrated in research as early as the 1980s and has been refined over decades. The basic idea uses an array of Mach-Zehnder interferometers (MZIs) — a specific type of optical device that splits a light signal into two paths, applies a controlled phase shift to one path, and recombines them. By choosing the phase shift, the MZI can implement an arbitrary 2×2 unitary transformation on the input light.

A grid of MZIs arranged in a specific topology (the Reck or Clements scheme) can implement any N×N unitary matrix. The input vector is encoded as the amplitudes of N wavelengths or N parallel waveguides, the matrix is configured by setting the phase shifts on each MZI, and the output is read by detectors at the other end. The whole computation happens at the speed of light through the grid — typically in the tens of picoseconds for an 8×8 or 16×16 matrix.

For practical AI workloads, larger matrices are needed than can be implemented in a single MZI grid (which is limited by chip area and crosstalk to roughly 64×64). Larger matrices are decomposed into tiles, with each tile computed by an MZI grid and the results combined. The tiling overhead reduces the per-operation efficiency, but the photonic compute step itself remains very fast.

## 4. The Challenges of Photonic Computing

Photonic computing has fundamental advantages but also significant challenges.

### 4.1 Conversion Costs

Information enters and leaves a photonic computer in electrical form. Inputs come from electronic memory, computations between layers may need to happen on conventional silicon, and outputs go back to electronic systems. Each transition between electrical and optical domains requires a digital-to-analog converter (DAC) and modulator on the input side, and a photodetector and analog-to-digital converter (ADC) on the output side. These conversion steps consume energy and add latency.

For workloads where the optical computation dominates the conversion costs — large matrix multiplications, where the photonic operation amortizes the conversion overhead — photonics wins. For workloads with frequent electrical-optical transitions and small per-step computation, the conversion costs eat the advantage. This is why photonic accelerators tend to be best at large-batch matrix multiplications and worst at small operations with high control flow overhead.

### 4.2 Precision

Photonic computation is inherently analog. The amplitudes and phases of light are continuous values, and the result of an interferometric computation is the actual physical interference pattern, which is read by an ADC. ADC precision is limited by noise, thermal effects, and the physics of the photodetector. Most current photonic accelerators provide 6-8 bits of effective precision, which is sufficient for inference of quantized neural networks but insufficient for training.

This is acceptable for AI inference (which already uses INT8 or FP8 for production) but rules out training applications where higher precision is needed. The companies in this space target inference exclusively for now, leaving training to electronic accelerators.

### 4.3 Manufacturing

Photonic chips are fabricated using silicon photonics processes, which are derived from but distinct from standard CMOS manufacturing. The silicon photonics ecosystem is much smaller than the CMOS ecosystem, with fewer foundries, fewer design tools, and less mature processes. Yields are lower and costs are higher. Companies in this space typically partner with specialized foundries (TSMC, GlobalFoundries, IMEC) to access silicon photonics processes, but the supply chain is fragile.

The integration of photonic devices with electronic devices on the same package — required for any practical accelerator — is itself a manufacturing challenge. The dies have different thermal characteristics, different electrical interfaces, and different reliability profiles. Co-packaging photonic and electronic components requires custom interposer and packaging technology that few foundries offer.

### 4.4 Programming Model

Conventional accelerators program through CUDA, ROCm, or similar high-level abstractions. Photonic accelerators have no equivalent maturity. Each vendor has its own programming model, often based on custom compilers that map TensorFlow or PyTorch operations onto the photonic hardware. The lack of standardization means that switching vendors requires rewriting software, and the lack of mature tooling means that bringing up a new model on photonic hardware is a significant undertaking.

## 5. Companies and Products

### 5.1 Lightmatter

Lightmatter, founded at MIT in 2017, is the most visible photonic AI startup. Its first product, Envise, is a photonic accelerator card designed for transformer inference. Envise combines photonic matrix multiplication with electronic memory and control logic on a single package. The company has demonstrated several models running on Envise hardware, including BERT and GPT-class transformers, with latency advantages over conventional GPUs for specific operations.

Lightmatter's second product, Passage, is a photonic interconnect designed to replace electrical chip-to-chip communication. Passage targets the data-movement problem rather than the computation problem, with the goal of letting conventional electronic accelerators communicate with each other at much higher bandwidth and lower latency than copper allows. Passage has gained more traction commercially than Envise, with announced partnerships with multiple chip vendors.

### 5.2 Lightelligence

Lightelligence (founded by Yichen Shen, also from MIT) is Lightmatter's main competitor. Its PACE (Photonic Arithmetic Computing Engine) is a photonic chip targeting matrix multiplication for inference workloads. The company has demonstrated 64×64 matrix multiplication at sub-nanosecond latency and has developed tooling for mapping CNN and transformer workloads onto its hardware. Lightelligence has attracted significant investment from Asian semiconductor companies and is building production-scale facilities in China.

### 5.3 Ayar Labs

Ayar Labs is focused entirely on the optical interconnect problem. Its TeraPHY chiplet provides optical I/O for conventional electronic chips, allowing them to communicate over fiber at terabit-per-second rates with very low energy per bit. Ayar Labs has partnerships with Intel, NVIDIA, and HPE, and its technology is being designed into next-generation AI systems for chip-to-chip and chip-to-memory communication.

Unlike Lightmatter and Lightelligence, Ayar Labs is not trying to perform computation in light — only to move data with it. This is a more conservative bet, but also one that fits more naturally into the existing chip ecosystem. The conventional chip vendors keep their compute architectures unchanged but get faster, lower-power interconnects.

### 5.4 Q.ANT

Q.ANT is a German photonics startup spun out of TRUMPF. Its photonic processor uses a different approach to interferometric computation, with a focus on very large-scale matrix operations. Q.ANT has secured EU funding and is positioning itself as a European alternative to the US-dominated AI hardware market.

### 5.5 PsiQuantum and Quantum-Adjacent Plays

PsiQuantum is technically a quantum computing company, not an AI accelerator company, but its silicon photonics manufacturing capabilities are relevant. The company has invested heavily in scaling silicon photonics production for its quantum computer, and the same manufacturing capabilities could be applied to photonic AI accelerators. Several of the technical challenges (precision, control, integration) are shared between photonic AI and photonic quantum computing.

## 6. Where Photonic Accelerators Fit

The strengths of photonic computing — energy-efficient large matrix multiplication, very fast data movement, parallel processing — map well onto specific subproblems in AI but poorly onto others. The most likely first commercial applications are:

**Large transformer inference**. The dominant operation is matrix multiplication, the precision requirements are modest (post-quantization), and the energy savings are most valuable for hyperscale deployments. This is the target market for Lightmatter Envise and Lightelligence PACE.

**Edge AI with strict power budgets**. Battery-powered devices, automotive AI, and other edge applications can benefit from the energy efficiency even if absolute throughput is lower than a GPU.

**Optical interconnect for conventional accelerators**. The Ayar Labs approach — replacing electrical interconnects with optical ones, while leaving the compute architecture unchanged — is the most pragmatic path and is most likely to see widespread adoption first.

**Recommendation systems**. Large embedding lookups followed by matrix multiplications are a good fit for photonic matrix engines.

The poor fits are equally important:

**Training**. Precision requirements rule out current photonic accelerators.
**Workloads with heavy control flow**. Conditional execution, dynamic shapes, and irregular memory access patterns do not benefit from photonic parallelism.
**Small or sparse operations**. The conversion overhead per operation dominates if the operation is small.

## 7. The Energy Argument

The strongest case for photonic AI is energy. AI inference is on track to consume meaningful fractions of global electricity production, and the per-token energy cost of frontier models is a growing economic and environmental concern. A 10× improvement in inference energy efficiency would be transformative for the economics of LLM deployment, and photonic accelerators have a credible path to this improvement.

The path is not straightforward. Photonic compute is more efficient per operation, but the conversion costs and the immature ecosystem currently eat much of the advantage. As manufacturing matures and conversion overhead drops, the gap between photonic and electronic energy efficiency should grow. The question is whether the maturity curve catches up to electronics' steady improvements before the funding for photonic research runs out.

## 8. The Realistic Timeline

In 2026, photonic accelerators are at roughly the stage that electronic AI accelerators were in 2014 — promising research demonstrations, a few early products, no widespread deployment. The first generation of products solves specific niche problems and provides existence proofs for the underlying technology. The next generation will need to expand the addressable workloads and bring down the unit economics. The third generation, if it arrives, may compete directly with electronic accelerators on broader workloads.

Most observers expect commercial photonic AI accelerators to remain niche through the late 2020s. The likely timeline is:

- **2026-2027**: First commercial deployments in inference applications. Limited to specific workloads where the technology has clear advantages.
- **2027-2029**: Broader product lines targeting more workloads. Photonic interconnects (Ayar Labs and similar) become standard in high-end AI systems.
- **2030+**: Possible mainstream adoption if the technology matures enough to compete on broad workloads. This is the optimistic scenario; the pessimistic one has photonic AI remaining a niche for the entire decade.

## 9. Why Hyperscalers Are Watching

Despite the uncertainty, the major cloud providers are actively investing in photonic technologies. Google has acquired multiple silicon photonics teams over the past several years. Microsoft has internal photonic computing research projects. Meta has invested in optical interconnect technologies for its data centers. AWS has not announced a photonic strategy but is presumed to have one.

The motivation is straightforward: AI workloads are growing faster than Moore's Law, and conventional electronic improvements are not keeping pace. If a 10× efficiency improvement is possible from any direction, the hyperscalers want to be in a position to exploit it. Photonics is one of the few directions with a credible path to such an improvement, and the cost of being wrong is small relative to the cost of being late.

## 10. Conclusion

Photonic AI accelerators are not yet a major part of the AI hardware landscape, and they may never be. The technical challenges are real, the manufacturing ecosystem is immature, and the conventional electronic accelerators are improving fast enough to make the photonic value proposition uncertain. But the underlying physics is real, the energy advantages are large, and the pace of progress in silicon photonics manufacturing has accelerated meaningfully since 2020.

For practitioners, the relevant question is whether to invest in software portability between conventional and photonic accelerators. The answer for most teams is no — the tooling is too immature, the products are too niche, and the abstraction layers needed to hide the difference do not exist. For research teams and those at the cutting edge of AI infrastructure, photonic accelerators are worth tracking. The first product to deliver on the energy promise at a price competitive with conventional GPUs will reshape the inference economics of frontier models, and that product may arrive sooner than the conservative timelines suggest.
