# Carbon Footprint and Sustainability in AI

*April 2026*

## 1. Introduction

The rapid expansion of large language models has created an energy and environmental challenge of unprecedented scale in the computing industry. Training a single frontier model can consume as much electricity as a small town uses in a year. The global fleet of AI inference servers runs continuously, consuming gigawatts of power. Data centers built to house AI hardware require millions of gallons of water for cooling. The semiconductor manufacturing process that produces AI accelerators is itself energy-intensive and resource-consuming.

These environmental costs are not abstract future concerns. They are present-day realities that are shaping corporate strategy, government policy, and public discourse about AI. Microsoft's 2024 sustainability report revealed that its carbon emissions had increased 29% since 2020, driven primarily by data center construction for AI workloads—directly threatening the company's pledge to be carbon negative by 2030. Google reported a 48% increase in greenhouse gas emissions over the same period, for similar reasons. The tension between AI scaling and sustainability commitments is now a central challenge for the largest technology companies.

This report provides a comprehensive technical examination of the environmental footprint of large language models. It covers the energy consumption of training and inference, carbon emissions, water usage, hardware lifecycle impacts, and the strategies—both technical and organizational—for mitigating these impacts. The intended audience is engineers, policy makers, and leaders who need to understand the environmental dimensions of AI deployment and make informed decisions about sustainability.

## 2. Energy Consumption of Training

### 2.1 Measuring Training Energy

The energy consumed by a training run is determined by three factors: the power draw of the hardware, the duration of the training run, and the overhead of the supporting infrastructure (cooling, networking, storage).

**Hardware power draw.** A single NVIDIA H100 GPU draws approximately 700 watts at full utilization. A training cluster of 16,000 H100s draws approximately 11.2 megawatts (MW) from the GPUs alone. Adding networking switches, CPUs, storage servers, and power conversion losses typically doubles the total facility power draw to 20-25 MW.

**Training duration.** Frontier model training runs last weeks to months. Llama 3 405B was trained for approximately 54 days on a cluster of 16,384 H100 GPUs. GPT-4's training is estimated to have taken 90-100 days on a comparable cluster.

**Total energy.** The product of power and time gives total energy. For a 20 MW facility running for 90 days:

```
20 MW × 90 days × 24 hours/day = 43,200 MWh = 43.2 GWh
```

This is approximately the annual electricity consumption of 4,000-5,000 average US households.

### 2.2 Specific Training Energy Estimates

Published and estimated training energy for notable models:

**Llama 3 405B (Meta, 2024).** Meta disclosed that Llama 3 training used 30.84 million GPU-hours on H100 GPUs. At approximately 700W per GPU, this translates to:

```
30.84M GPU-hours × 700W = 21.6 GWh (GPU only)
```

Including infrastructure overhead (PUE of ~1.1-1.2), total facility energy was approximately 24-26 GWh.

**GPT-4 (OpenAI, 2023).** Not officially disclosed. Estimates based on cluster size and training duration range from 40 to 80 GWh. The wide range reflects uncertainty about the model architecture, cluster size, and training duration.

**Gemini Ultra (Google, 2023).** Not officially disclosed. Google's training infrastructure uses TPUs rather than GPUs, with different power characteristics. Estimates range from 20 to 50 GWh.

**Llama 2 70B (Meta, 2023).** Meta disclosed 1,720,320 GPU-hours on A100 GPUs. At approximately 400W per GPU:

```
1.72M GPU-hours × 400W = 688 MWh ≈ 0.69 GWh
```

Substantially less than frontier models trained a year later, reflecting the exponential growth in training compute.

**Smaller models for comparison:** Training a 7B model on 1T tokens requires approximately 0.5-1 GWh—roughly the annual energy consumption of 50-100 households. This illustrates how even "small" large models have meaningful energy footprints.

### 2.3 The Training Energy Trajectory

Training energy for frontier models has increased by approximately 4-5x per year from 2020 to 2025, driven by increases in both model size and training data. This exponential growth substantially outpaces the improvement in hardware energy efficiency (approximately 2x per generation, or ~2x per 2 years), meaning that the total energy consumption of frontier training runs is increasing in absolute terms.

Extrapolating this trend, frontier training runs in 2027-2028 could consume hundreds of GWh—approaching the output of a small power plant dedicated to a single training run. This is the "energy wall" discussed in scaling laws literature, and it is already influencing the physical infrastructure decisions of frontier AI labs.

### 2.4 Failed Runs and Overhead

Published training energy figures typically represent the successful training run. In practice, reaching that successful run requires:

**Hyperparameter search.** Preliminary runs to determine learning rates, batch sizes, data mixtures, and other hyperparameters. These consume 5-20% of the final training run's compute.

**Failed runs.** Training instabilities, hardware failures, data quality issues, and other problems can cause training runs to fail partway through and be restarted. For complex training recipes, the total compute including failures can be 1.5-3x the compute of the successful run.

**Checkpoint evaluation.** Periodically evaluating model checkpoints on validation sets and benchmarks consumes additional compute.

The total energy cost of developing a frontier model—including all experimentation, failed runs, and evaluation—is typically 2-5x the energy of the final successful training run alone.

## 3. Energy Consumption of Inference

### 3.1 Per-Query Energy

The energy consumed by a single LLM inference query depends on the model size, the number of input and output tokens, and the hardware:

**A single GPT-4-class query** (1,000 input tokens, 500 output tokens) on H100 hardware consumes approximately:

```
Prefill: ~0.5 seconds at ~700W = ~0.1 Wh
Decode: ~5 seconds at ~700W = ~1.0 Wh
Total: ~1.1 Wh per query
```

For comparison, a Google search query consumes approximately 0.3 Wh (as estimated by Google in 2009; current estimates are higher with AI-enhanced results), and a standard web page load consumes approximately 0.01-0.1 Wh.

**A smaller model query** (e.g., GPT-4o-mini-class, 1,000 input tokens, 500 output tokens) consumes roughly 5-10x less energy: approximately 0.1-0.2 Wh per query.

### 3.2 Fleet-Level Energy

The aggregate energy consumption of LLM inference is driven by query volume. Estimated inference volumes for major providers (as of early 2026):

- **ChatGPT:** ~200-500 million queries per day
- **Google Gemini (in Search and other products):** ~1-10 billion queries per day (including AI-enhanced search results)
- **Anthropic Claude:** ~50-200 million queries per day
- **Microsoft Copilot (across products):** ~500 million-2 billion queries per day

Even at conservative estimates, the global LLM inference fleet processes billions of queries per day. At an average of 0.5 Wh per query (blending large and small models), this translates to:

```
5 billion queries/day × 0.5 Wh = 2.5 GWh/day = 912 GWh/year
```

This is comparable to the electricity consumption of a small country. The actual figure is likely higher, because this estimate excludes enterprise API usage, self-hosted deployments, and the many smaller providers serving open-weight models.

### 3.3 Inference Dominates Training

For widely deployed models, inference energy consumption quickly exceeds training energy consumption. If a model costs 50 GWh to train and serves 500 million queries per day at 0.5 Wh per query, the inference energy is:

```
500M × 0.5 Wh = 250 MWh/day
```

The training energy is matched in approximately 200 days (~7 months) of inference. Over the model's lifetime of 1-2 years, inference energy exceeds training energy by 3-10x. This has an important implication for sustainability: optimizing inference efficiency has a larger total impact than optimizing training efficiency, because inference runs continuously at scale.

### 3.4 The Inference Energy Trajectory

Two opposing forces shape the inference energy trajectory:

**Increasing demand.** As LLMs are integrated into more products and used by more people, total query volume grows rapidly. New applications (AI-enhanced search, code completion, document processing, customer support) each add substantial incremental query volume.

**Increasing efficiency.** Hardware improvements, quantization, better serving frameworks, model distillation, and architectural innovations reduce the energy per query. The deflation rate is approximately 3-4x per year for a given quality level (as discussed in the companion article on LLM economics).

The net effect depends on whether efficiency improvements outpace demand growth. Current evidence suggests that demand growth is outpacing efficiency gains, leading to net increases in total inference energy consumption. This is the "rebound effect" discussed in Section 10.

## 4. Carbon Emissions

### 4.1 From Energy to Carbon

The carbon footprint of AI energy consumption depends on the carbon intensity of the electricity grid powering the data centers. Carbon intensity varies enormously by region and time:

- **Norway (hydro-dominant):** ~20 gCO2/kWh
- **France (nuclear-dominant):** ~60 gCO2/kWh
- **US average:** ~400 gCO2/kWh
- **US (Virginia, coal + gas heavy):** ~350-450 gCO2/kWh
- **US (Oregon, hydro):** ~100-150 gCO2/kWh
- **US (Iowa, wind + natural gas):** ~250-350 gCO2/kWh
- **Germany (mixed with coal):** ~350 gCO2/kWh
- **India (coal-heavy):** ~700 gCO2/kWh
- **China (coal-heavy):** ~550 gCO2/kWh

A training run consuming 50 GWh in a region with 400 gCO2/kWh produces:

```
50,000 MWh × 400 kg CO2/MWh = 20,000 tonnes CO2
```

This is equivalent to the annual emissions of approximately 4,000 cars. The same training run in Norway (20 gCO2/kWh) would produce only 1,000 tonnes CO2—a 20x difference based solely on location.

### 4.2 Marginal vs. Average Carbon Intensity

The average carbon intensity of a grid reflects the mix of generation sources (coal, gas, nuclear, hydro, wind, solar) averaged over the year. But when a data center increases its electricity demand, the question is which generator is dispatched to meet that incremental demand—and that marginal generator is often a fossil fuel plant, even on grids with substantial renewable capacity.

This means that using average carbon intensity may understate the actual carbon impact of AI workloads. If a data center in Oregon draws more power during peak hours, the marginal generation may come from a natural gas plant (at ~500 gCO2/kWh) rather than the hydro dams (at ~20 gCO2/kWh) that provide most of the grid's average supply.

The correct methodology for carbon accounting is debated. Using average intensity provides a simpler, more standardized measure. Using marginal intensity provides a more accurate picture of the actual emissions caused by the workload. Most corporate carbon reporting uses average intensity, which tends to understate the actual impact.

### 4.3 Scope 1, 2, and 3 Emissions

Following the Greenhouse Gas Protocol, AI-related emissions are categorized:

**Scope 1 (direct emissions).** Emissions from sources owned by the company—primarily backup diesel generators at data centers. These are typically small relative to Scope 2.

**Scope 2 (electricity emissions).** Emissions from purchased electricity used to power data centers. This is the dominant category for AI operations and is the focus of most analyses.

**Scope 3 (supply chain emissions).** Emissions from the manufacture of GPUs, servers, networking equipment, and data center construction. This includes the carbon emitted during semiconductor fabrication, the mining of raw materials, and the transportation of equipment. Scope 3 emissions are less frequently measured but can be substantial—estimates suggest that the embodied carbon of a modern GPU is 50-150 kgCO2, and a training cluster of 16,000 GPUs represents 800-2,400 tonnes CO2 in embodied carbon alone.

## 5. Water Consumption

### 5.1 Cooling Water

Data centers generate enormous amounts of heat, and many use evaporative cooling systems that consume water. The water usage effectiveness (WUE) metric measures the liters of water consumed per kilowatt-hour of energy used:

- **Air-cooled data centers:** WUE ≈ 0 (no water consumption for cooling, but less energy-efficient)
- **Evaporative cooling:** WUE ≈ 1.0-2.0 L/kWh
- **Hybrid systems:** WUE ≈ 0.5-1.0 L/kWh

For a data center consuming 20 MW with a WUE of 1.5 L/kWh:

```
20 MW × 24 hours × 1.5 L/kWh = 720,000 liters/day ≈ 190,000 gallons/day
```

Over a year, this is approximately 69 million gallons—roughly the water consumption of a small agricultural farm.

### 5.2 Microsoft and Google Water Disclosures

Microsoft reported that its global water consumption increased by 34% in 2022 compared to 2021, attributing the increase partly to AI infrastructure expansion. Google reported a 20% increase in water consumption over the same period. Both companies have set water replenishment targets—pledging to replenish more water than they consume by 2030—but achievement of these targets is uncertain given the pace of AI infrastructure growth.

### 5.3 Water Stress

The environmental impact of water consumption depends on local water stress—the ratio of water demand to available supply. A data center consuming 190,000 gallons per day in a water-rich region (e.g., the Pacific Northwest) has minimal environmental impact. The same consumption in a water-stressed region (e.g., Phoenix, Arizona, or parts of India) can contribute to local water scarcity.

Several major data center hubs are in water-stressed regions. The Dalles, Oregon (home to Google's major data center campus) has experienced tensions between the data center's water demand and local water resources. Phoenix and the broader Southwest face severe water stress, yet continue to attract data center construction due to land availability and favorable tax incentives.

## 6. Hardware Lifecycle and Embodied Carbon

### 6.1 GPU Manufacturing

The embodied carbon of a GPU—the carbon emitted during its manufacture, from raw material extraction through fabrication to assembly—is a significant but often overlooked component of AI's carbon footprint.

Semiconductor fabrication is extremely energy-intensive. A modern GPU die is manufactured in a fabrication facility (fab) that consumes 100+ MW of electricity continuously. The fab process involves hundreds of steps over several months, including photolithography, etching, deposition, and testing. The energy consumed per GPU die is estimated at 50-200 kWh, depending on the process node and die size.

In addition to the die, a GPU package includes HBM memory stacks (each with their own fabrication energy), a substrate, and packaging materials. The total embodied carbon of an NVIDIA H100 GPU is estimated at 50-150 kgCO2, though NVIDIA has not published official lifecycle assessment data.

### 6.2 Server and Infrastructure Manufacturing

Beyond GPUs, the rest of the server (CPUs, motherboards, memory, SSDs, power supplies, chassis), networking equipment (switches, cables, optics), and data center construction materials (steel, concrete, copper for power distribution) all have embodied carbon.

A complete 8× H100 server (DGX H100-class) has an estimated embodied carbon of 2,000-5,000 kgCO2, with the GPUs representing 30-50% of the total and the HBM memory representing another 20-30%.

### 6.3 Hardware Lifetime and Replacement

AI hardware has a typical useful lifetime of 3-5 years before being replaced by newer, more efficient hardware. The short replacement cycle means that embodied carbon is amortized over a relatively short period, increasing the per-year carbon intensity of embodied emissions.

When hardware is replaced, responsible disposal requires recycling of precious metals (gold, palladium, rare earth elements), proper handling of hazardous materials, and data destruction for security. The e-waste from AI hardware is growing rapidly with the industry, and responsible disposal practices are an increasing concern.

### 6.4 The Embodied vs. Operational Carbon Balance

For a typical AI data center, the balance between embodied carbon (hardware manufacturing and construction) and operational carbon (electricity) depends on the grid carbon intensity:

- **High-carbon grid (400+ gCO2/kWh):** Operational emissions dominate. Hardware embodied carbon represents 10-20% of total lifecycle emissions.
- **Low-carbon grid (50-100 gCO2/kWh):** Embodied carbon becomes proportionally more significant, potentially reaching 30-50% of total lifecycle emissions.
- **Zero-carbon grid (0 gCO2/kWh):** All emissions come from embodied carbon—hardware manufacturing, construction, and supply chain.

This has implications for sustainability strategy. Shifting to renewable electricity addresses operational emissions but does not eliminate embodied emissions. A comprehensive approach must address both.

## 7. Power Usage Effectiveness (PUE)

### 7.1 What PUE Measures

Power Usage Effectiveness (PUE) is the ratio of total facility power to IT equipment power:

```
PUE = Total Facility Power / IT Equipment Power
```

A PUE of 1.0 would mean all facility power goes directly to computing equipment—no overhead for cooling, lighting, power distribution, or other infrastructure. In practice:

- **Average data center:** PUE ≈ 1.5-1.6
- **Efficient hyperscale data center:** PUE ≈ 1.1-1.2
- **Best-in-class (Google):** PUE ≈ 1.06-1.10
- **Liquid-cooled AI clusters:** PUE ≈ 1.03-1.08

### 7.2 AI-Specific PUE Challenges

AI hardware presents specific challenges for PUE:

**High power density.** A rack of 8× H100 GPUs draws 10-15 kW, and denser configurations (like NVIDIA's DGX GB200 NVL72) can draw 120+ kW per rack. This extreme power density generates concentrated heat that is difficult to remove with traditional air cooling, pushing toward liquid cooling solutions.

**Liquid cooling.** Direct liquid cooling (where coolant flows through heat sinks attached directly to GPUs) is becoming standard for AI clusters. Liquid cooling is more efficient than air cooling (lower PUE) but requires additional infrastructure—piping, coolant distribution units, and leak detection systems—that adds to the capital cost of the facility.

**Variable load.** Training runs are continuous high-power loads, but inference loads may vary throughout the day. Cooling systems optimized for peak load are inefficient at partial load, potentially increasing PUE during off-peak hours.

### 7.3 PUE Limitations

PUE measures energy efficiency but not carbon efficiency. A data center with PUE 1.1 powered by coal has a much larger carbon footprint than a data center with PUE 1.3 powered by renewables. Carbon Usage Effectiveness (CUE = total carbon emissions / IT equipment energy) is a more relevant metric for sustainability assessment but is less widely reported.

## 8. Corporate Sustainability Efforts

### 8.1 Google

Google has claimed to be carbon-neutral for its operations since 2007, achieved through a combination of renewable energy purchases and carbon offsets. In 2020, Google announced a more ambitious goal: operating on carbon-free energy 24/7 at all data centers by 2030. This means matching energy consumption with carbon-free generation not just on an annual average basis (which allows fossil fuel use at night, offset by solar overproduction during the day) but on an hourly basis.

Google's AI energy consumption growth is straining this commitment. The 2024 environmental report showed a 48% increase in greenhouse gas emissions since 2019, driven by data center expansion. Google acknowledges that achieving 24/7 carbon-free energy is more challenging than anticipated, given the pace of AI infrastructure buildout.

Google's sustainability approach for AI includes:

- **TPU efficiency.** Google's custom TPUs are designed for energy efficiency on AI workloads, and Google claims they are 2-3x more energy-efficient per FLOP than comparable GPUs for the workloads they target.
- **Renewable energy procurement.** Google has signed long-term power purchase agreements (PPAs) for renewable energy, including wind and solar farms near data center locations.
- **Carbon-intelligent computing.** Google shifts non-urgent workloads (including some AI training) to data centers and times with lower carbon intensity, reducing the carbon impact without reducing computation.
- **Geothermal energy.** Google has invested in next-generation geothermal energy (Enhanced Geothermal Systems) near its Nevada data center, aiming for 24/7 carbon-free baseload power.

### 8.2 Microsoft

Microsoft pledged in 2020 to be carbon negative by 2030 and to remove all historical carbon emissions by 2050. The company's AI infrastructure expansion has made this commitment significantly more difficult.

Microsoft's approach includes:

- **Nuclear energy.** Microsoft signed a 20-year agreement with Constellation Energy to restart a unit at Three Mile Island nuclear power plant, providing 835 MW of carbon-free baseload power.
- **Small modular reactors (SMRs).** Microsoft has explored agreements with SMR developers for future data center power.
- **Carbon removal investments.** Microsoft has invested in carbon removal technologies (direct air capture, enhanced weathering) to offset emissions it cannot eliminate. The company has contracted for over 5 million tonnes of carbon removal.
- **Internal carbon fee.** Microsoft charges its business units an internal carbon fee ($100/tonne for Scope 1 and 2 emissions), creating a financial incentive to reduce emissions.

### 8.3 Meta

Meta committed to reaching net zero emissions across its value chain by 2030. The company's approach to AI sustainability includes:

- **100% renewable energy.** Meta has matched its global electricity consumption with 100% renewable energy purchases since 2020.
- **Open model efficiency research.** Meta's publication of Llama models and training details enables the broader community to identify and implement efficiency improvements.
- **Training efficiency disclosure.** Meta's Llama 3 technical report included detailed energy and emissions estimates, setting a precedent for transparency in training environmental impact.

### 8.4 Anthropic, OpenAI, and Smaller Labs

Smaller AI labs have generally disclosed less about their environmental impact:

- **Anthropic** has not published detailed sustainability reports but has stated its commitment to responsible AI development, which includes environmental considerations.
- **OpenAI** has not published detailed energy or carbon data for its training runs. Some sustainability information is included in Microsoft's reporting (since OpenAI uses Microsoft Azure infrastructure).

The lack of standardized reporting requirements means that the environmental impact of AI development is difficult to assess comprehensively across the industry.

## 9. Carbon Accounting Frameworks

### 9.1 The Challenge of AI Carbon Accounting

Accounting for the carbon impact of AI workloads is complicated by several factors:

**Shared infrastructure.** AI workloads share data centers with other workloads. Attributing the facility's emissions to specific workloads requires allocation methodologies that may be arbitrary.

**Renewable energy certificates (RECs).** Companies can purchase RECs that represent renewable generation elsewhere on the grid, effectively claiming credit for renewable energy that may not physically power their data center. The environmental value of RECs is debated—they provide financial incentives for renewable development but do not guarantee that the data center's actual electricity is carbon-free.

**Carbon offsets.** Companies can purchase carbon offsets (representing avoided emissions or carbon removal elsewhere) to claim net-zero or carbon-negative status. The quality and permanence of carbon offsets vary widely, and there is growing skepticism about whether offsets represent genuine emissions reductions.

**Supply chain complexity.** Scope 3 emissions from hardware manufacturing involve a global supply chain spanning semiconductor fabs in Taiwan, memory manufacturing in South Korea, assembly in China, and numerous component suppliers. Tracking emissions through this supply chain is extremely difficult.

### 9.2 Emerging Standards

Several frameworks are being developed for AI carbon accounting:

**ML CO2 Impact.** A tool and methodology for estimating the carbon footprint of ML training, based on hardware type, training duration, and grid carbon intensity. Provides a standardized way to report training emissions.

**Green AI.** A research initiative promoting the reporting of computational cost alongside accuracy in ML research papers. The goal is to make efficiency a first-class evaluation criterion, not an afterthought.

**ISO 14064 and the GHG Protocol.** Existing corporate carbon accounting standards apply to AI workloads but do not provide AI-specific guidance. Work is underway to develop supplementary guidance for digital infrastructure and AI-specific emissions.

**EU Corporate Sustainability Reporting Directive (CSRD).** The EU's mandatory sustainability reporting requirements, which take effect in stages from 2024 to 2028, require companies to report detailed carbon emissions data, including for data center operations. This will increase transparency about the environmental impact of AI operations in Europe.

### 9.3 Model Cards and Environmental Disclosure

A growing practice is to include environmental impact information in model cards—the standardized documentation that accompanies model releases:

- **Training energy consumption** (in GPU-hours and estimated kWh)
- **Carbon emissions** (estimated tonnes CO2, with assumptions about grid intensity)
- **Hardware used** (GPU type, cluster size, cloud region)
- **Training efficiency** (MFU, tokens per GPU-hour)

Meta's Llama 3 model card set a positive precedent by including relatively detailed energy and emissions estimates. Standardizing such disclosures across the industry would significantly improve transparency.

## 10. Efficiency Improvements and Rebound Effects

### 10.1 Sources of Efficiency Improvement

Multiple factors are reducing the energy consumed per unit of AI output:

**Hardware efficiency.** Each GPU generation provides ~2x improvement in performance per watt. The transition from A100 to H100 to B200 has improved energy efficiency for LLM inference by roughly 4-6x over 4 years.

**Algorithmic efficiency.** FlashAttention reduces the memory and compute requirements for attention by 2-4x. Better optimizers (AdamW → Lion → Sophia) reduce training FLOPs by 10-30%. Mixture-of-experts architectures reduce compute per token by 2-8x. These algorithmic improvements compound hardware improvements.

**Quantization.** Running models at INT4 or FP8 instead of FP16 reduces energy consumption per inference by 2-4x, because lower precision requires fewer memory reads and simpler arithmetic operations.

**Model distillation.** Training smaller models to match larger model performance enables serving equivalent quality with 5-50x less energy per query.

**Software optimization.** Better inference serving frameworks (vLLM, TensorRT-LLM), continuous batching, and speculative decoding improve throughput per GPU by 2-5x relative to naive implementations.

The combined effect of these improvements is dramatic: the energy required to achieve GPT-3.5-level performance has decreased by approximately 50-100x between 2022 and 2026.

### 10.2 The Rebound Effect

The Jevons paradox (or rebound effect) holds that improvements in the efficiency of resource use tend to increase rather than decrease total resource consumption, because the reduced cost makes the resource available for new uses. This applies directly to AI:

**More queries.** As inference becomes cheaper, more applications become viable, more users gain access, and query volumes increase. The cost reduction enables new use cases (AI-enhanced search, continuous code completion, real-time document processing) that would not be economical at higher per-query costs.

**Larger models.** Efficiency improvements in hardware and software are used to train and serve larger, more capable models rather than to reduce energy consumption. Each generation of hardware enables a larger model, not a cheaper run of the same-size model.

**New modalities.** As energy efficiency improves for text models, developers deploy models for new modalities—image generation, video generation, speech synthesis—that consume additional energy. Generating a single AI image (through diffusion models) can consume 10-100x the energy of a text query.

Empirically, the rebound effect appears to be more than 100% for AI—meaning that total energy consumption is increasing despite dramatic efficiency improvements per query. This pattern is not unique to AI (it mirrors the history of computing more broadly) but it has specific implications for sustainability planning: efficiency improvements alone will not reduce total energy consumption if demand continues to grow.

### 10.3 Efficiency vs. Demand: The Race

The question of whether AI's total energy consumption will grow, stabilize, or decline depends on the balance between efficiency improvements and demand growth:

**Short term (2026-2028).** Total energy consumption is almost certainly increasing. The rate of AI adoption and the growth in query volumes are outpacing efficiency improvements. Major infrastructure build-outs (new data centers, new GPU clusters) are committed and will consume energy regardless of efficiency gains.

**Medium term (2028-2032).** The trajectory is uncertain. If the scaling paradigm reaches diminishing returns (the "scaling wall"), demand growth may slow. If efficiency improvements continue at current rates, per-query energy could fall enough to offset moderate demand growth. But if new AI modalities (robotics, continuous agents, real-time video processing) create large new sources of demand, total consumption could continue to increase rapidly.

**Long term (2032+).** Deeply uncertain. The outcome depends on whether AI development follows the path of previous computing technologies (where efficiency improvements eventually stabilized consumption) or whether AI represents a fundamentally different energy demand curve.

## 11. Water and Cooling Innovations

### 11.1 Liquid Cooling Adoption

The shift from air cooling to liquid cooling for AI clusters is driven by necessity (GPU power density exceeds what air cooling can handle) rather than environmental considerations. However, liquid cooling has environmental benefits:

**Reduced PUE.** Liquid cooling is more efficient than air cooling, reducing the overhead power consumed by the cooling system. This can reduce PUE from 1.3-1.5 (air-cooled) to 1.03-1.10 (liquid-cooled), saving 15-30% of total facility energy.

**Reduced water consumption.** Direct liquid cooling systems can reject heat through dry coolers (air-based heat rejection) rather than evaporative cooling, reducing or eliminating water consumption. Some liquid cooling systems use completely sealed coolant loops with zero water consumption.

**Waste heat recovery.** Liquid cooling captures heat at higher temperatures (40-60°C) than air cooling (25-35°C), making the waste heat more useful for district heating, industrial processes, or other applications. Several data centers in Nordic countries sell waste heat to district heating networks, turning an environmental cost into an environmental benefit.

### 11.2 Cold Climate Locations

Locating data centers in cold climates reduces cooling energy requirements:

- **Nordic countries (Sweden, Finland, Norway, Iceland).** Cold ambient temperatures enable free cooling for much of the year, reducing or eliminating mechanical cooling. Combined with low-carbon electricity (hydro and wind), Nordic locations offer the lowest-carbon AI infrastructure.
- **Canada.** Similar climate advantages, combined with hydroelectric power in Quebec and British Columbia.
- **Northern US.** States like Iowa, Oregon, and Washington offer moderate climates and access to wind or hydro power.

The trade-off is network latency—data centers in remote northern locations may have higher latency to major population centers. For training (where latency does not matter), northern locations are ideal. For latency-sensitive inference, locations closer to users are preferred.

## 12. Regulatory Landscape

### 12.1 EU Regulation

The European Union has been the most active regulatory body on AI sustainability:

**Energy Efficiency Directive (EED).** Requires data centers above 500 kW to report energy consumption, PUE, renewable energy usage, and water consumption. This creates a transparency baseline for AI infrastructure in Europe.

**Corporate Sustainability Reporting Directive (CSRD).** Requires large companies operating in the EU to report detailed sustainability data, including Scope 1, 2, and 3 emissions. This will capture AI-related emissions for companies operating AI infrastructure in Europe.

**EU AI Act.** While primarily focused on safety and ethics, the EU AI Act includes provisions for environmental impact reporting for high-risk AI systems.

### 12.2 US Regulation

US federal regulation of AI sustainability is minimal as of 2026. However:

**State-level requirements.** Several states (California, Oregon, Virginia) have data center energy reporting requirements or renewable energy mandates.

**SEC climate disclosure rules.** The SEC's climate disclosure rules (if fully implemented) would require publicly traded companies to report material climate-related risks, which could include energy and carbon data for AI operations.

**Executive orders.** Executive Order 14110 on AI (October 2023) directed federal agencies to consider the environmental impact of AI development but did not establish binding requirements.

### 12.3 Voluntary Commitments

In the absence of comprehensive regulation, corporate voluntary commitments are the primary mechanism for AI sustainability:

- Science-Based Targets initiative (SBTi) membership
- RE100 (commitment to 100% renewable electricity)
- The Climate Pledge (net-zero by 2040)

These voluntary commitments are valuable for setting direction but lack enforcement mechanisms and are vulnerable to revision when they conflict with business priorities (as Microsoft's and Google's experiences demonstrate).

## 13. Practical Recommendations

### 13.1 For AI Infrastructure Operators

**Measure and report.** Implement energy monitoring at the GPU, server, and facility level. Report energy consumption and carbon emissions in a standardized format. Transparency is a prerequisite for improvement.

**Choose low-carbon locations.** When building or leasing data center capacity, prioritize locations with low-carbon electricity grids. The carbon intensity difference between a coal-heavy grid and a hydro/nuclear grid can be 10-20x, dwarfing any efficiency optimization.

**Procure renewable energy.** Sign long-term PPAs for renewable energy that is additional (i.e., that would not have been built without the PPA). Additionality ensures that the PPA drives new renewable generation rather than claiming credit for existing capacity.

**Invest in cooling efficiency.** Deploy liquid cooling for AI clusters, recover waste heat where feasible, and design for low PUE. The energy savings from efficient cooling compound over the lifetime of the facility.

**Plan for hardware lifecycle.** Establish responsible procurement practices that consider embodied carbon, and develop recycling and disposal programs for end-of-life hardware.

### 13.2 For AI Developers

**Optimize model efficiency.** Use quantization, distillation, and architecture search to minimize model size for a given quality target. A model that is 2x smaller and 2x faster is also 2x more energy-efficient per query.

**Use efficient training recipes.** Invest in learning rate schedules, data selection, and training techniques that achieve target quality with fewer FLOPs. A 20% improvement in training efficiency reduces both cost and energy by 20%.

**Report environmental impact.** Include energy consumption and carbon estimates in model cards and technical reports. Follow Meta's lead with the Llama 3 technical report.

**Consider inference optimization.** Since inference dominates lifetime energy consumption, invest in inference optimization (quantization, batching, caching, model routing) to reduce per-query energy.

### 13.3 For AI Users and Procurers

**Choose efficient models.** Use the smallest model that meets quality requirements. The energy difference between a small model and a large model is 10-50x per query.

**Choose low-carbon providers.** Where possible, select inference providers that operate on low-carbon grids and report their environmental impact.

**Cache and batch.** Cache responses for repeated queries and batch non-urgent queries, reducing total inference energy.

**Evaluate the energy trade-off.** Consider whether the AI-powered approach is genuinely more energy-efficient than the alternative. An AI system that replaces a manual process may use more or less energy than the manual process it replaces, depending on the specifics.

## 14. The Broader Context

### 14.1 AI Energy in Global Context

Global electricity generation is approximately 28,000 TWh per year. Current estimates of AI-related electricity consumption range from 50-150 TWh per year (including training and inference across all providers), or approximately 0.2-0.5% of global electricity. This is comparable to the electricity consumption of a small-to-medium country (e.g., the Netherlands at ~120 TWh/year).

While 0.2-0.5% of global electricity may seem modest, it is growing rapidly, and the concentrated nature of AI energy demand (in a relatively small number of data centers operated by a handful of companies) creates local impacts on grid capacity, electricity prices, and resource allocation.

### 14.2 The Opportunity Cost of AI Energy

Energy consumed by AI is energy not available for other uses. In regions with constrained grid capacity, AI data centers compete with residential, commercial, and industrial consumers for electricity. Northern Virginia, the world's largest data center market, has experienced grid capacity constraints that have delayed new data center construction and raised electricity prices for all consumers.

The opportunity cost argument is particularly relevant for developing countries, where AI energy consumption could divert scarce electricity from essential services like healthcare, education, and basic industry.

### 14.3 The Productivity Argument

Proponents of AI scaling argue that the energy consumed by AI is justified by the productivity improvements it enables. If AI-powered tools make knowledge workers 20% more productive, the economic value of that productivity improvement may exceed the cost (including environmental cost) of the energy consumed. This argument has merit but is difficult to quantify precisely and does not address the environmental externalities (carbon emissions, water consumption, habitat disruption) that are not reflected in market prices.

### 14.4 The Path Forward

The most plausible path to sustainable AI involves:

1. **Continued efficiency improvements** that reduce energy per unit of AI output.
2. **Rapid decarbonization of electricity grids** that reduces the carbon intensity of remaining energy consumption.
3. **Transparent measurement and reporting** that enables informed decision-making and accountability.
4. **Regulatory frameworks** that internalize environmental costs (through carbon pricing, energy efficiency standards, or reporting mandates).
5. **Research into fundamentally more efficient computing paradigms** (neuromorphic computing, optical computing, analog computing) that could reduce energy consumption by orders of magnitude for specific workloads.

None of these individually is sufficient. Efficiency improvements alone will not reduce total consumption if demand grows faster. Decarbonization alone does not address water consumption, embodied carbon, or e-waste. Regulation alone does not drive the technical innovation needed for step-change improvements. A comprehensive approach requires all of these elements working in concert.

## 15. Conclusion

The environmental footprint of large language models is substantial and growing. Training a single frontier model emits thousands of tonnes of CO2 and consumes tens of gigawatt-hours of electricity. The global inference fleet consumes hundreds of gigawatt-hours per year, a figure that is growing rapidly as AI is integrated into more products and used by more people. Data centers consume millions of gallons of water for cooling. The hardware lifecycle—from semiconductor fabrication to end-of-life disposal—adds embodied carbon and e-waste.

These environmental costs are not inherent to AI technology. They are the result of specific choices about how AI is developed and deployed: which energy sources power the data centers, how efficiently the hardware and software operate, where the facilities are located, and whether environmental costs are measured and managed. The same AI capabilities can be delivered with dramatically different environmental footprints depending on these choices.

The most important intervention is decarbonization of the electricity grid. The carbon intensity of electricity varies by 10-30x across regions, and the choice of data center location is the single largest determinant of AI's carbon footprint. Companies that are serious about sustainability will locate their most energy-intensive workloads in regions with clean electricity and invest in additional renewable generation to ensure that their demand drives new clean capacity rather than displacing existing users.

The second most important intervention is efficiency optimization. The tools are available today—quantization, distillation, efficient serving, model routing—to reduce energy consumption per query by 5-50x at a given quality level. These optimizations reduce both cost and environmental impact, aligning economic and environmental incentives.

The least certain but most important long-term question is whether the rebound effect will overwhelm efficiency gains, leading to ever-increasing total energy consumption, or whether AI energy demand will eventually stabilize as the technology matures and efficiency improvements catch up with demand growth. The answer to this question depends on choices made by the industry, regulators, and the public in the coming years. Measurement, transparency, and accountability are the prerequisites for making those choices wisely.
