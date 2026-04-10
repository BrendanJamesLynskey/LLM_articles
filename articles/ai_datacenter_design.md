# AI Datacenter Design: Power, Cooling, Networking, and the Grid

*April 2026 • Technical Report*

## 1. Introduction

A typical web service datacenter from the cloud computing era was designed around a simple set of constraints: pack as many servers as possible into a given area, provide enough power and cooling to keep them running, and connect them to the network. The economics were dominated by real estate, electricity, and capital amortization of the buildings and equipment. Power densities were modest — 5 to 15 kilowatts per server rack — and cooling was provided by computer room air conditioning units that pushed cold air through perforated floor tiles. The overall design was a refinement of office buildings: large open spaces, redundant power distribution, redundant cooling, redundant networking.

AI training and inference are not friendly to this design. A modern Blackwell-based server rack draws 120 kilowatts at peak load, more than ten times the density of a conventional server rack. Air cooling is physically incapable of removing this much heat from the small floor area of a rack. The power delivery, cooling distribution, networking topology, and even the building structure must be designed differently. The result is that purpose-built AI datacenters are emerging as a distinct facility type, with very different design constraints from the cloud datacenters they are gradually displacing.

The scale of investment is staggering. Microsoft, Meta, Google, Amazon, and OpenAI/Stargate have collectively committed over $500 billion to AI datacenter construction over the next several years. Individual sites are reaching 1 GW of power capacity — comparable to a small nuclear plant — and the next generation of designs is targeting 5 GW and beyond. The strain on electrical grids, water supplies, and local communities is substantial and growing.

This report examines the design principles of modern AI datacenters: the power systems, cooling technologies, networking topologies, and the broader interactions with the electrical grid and the physical environment.

## 2. The Power Challenge

### 2.1 Rack Density Explosion

The power density of a server rack has grown roughly 10× over the past decade, almost entirely driven by AI accelerators. A typical CPU-based server rack from 2015 drew 10 kW and contained perhaps 20 servers. A modern AI training rack — say, an NVIDIA NVL72 — draws 120 kW and contains 18 compute trays. The power per square meter of floor area has increased by a similar factor, fundamentally changing the design of the facility.

This growth shows no signs of stopping. The next-generation Rubin systems are projected to draw 200+ kW per rack. Custom designs from hyperscalers are exploring 300-500 kW per rack. The industry is heading toward a future where individual racks consume the power of entire conventional datacenters.

### 2.2 Power Delivery

Delivering tens of megawatts of power to a single building requires substantial electrical infrastructure. AI datacenters typically connect to the high-voltage grid (115 kV or higher) and step the voltage down through multiple transformer stages. The power distribution within the building uses busways and bus ducts rather than the cable trays of conventional datacenters, because the current capacities required exceed what cables can practically handle.

The power delivery system also has to be highly reliable. AI training jobs run for weeks or months, and an unplanned power loss can lose hours or days of work and cost millions of dollars. The standard approach uses uninterruptible power supplies (UPS) sized for several minutes of full-load operation, backed by diesel or gas generators that can sustain the facility indefinitely. Some new designs use battery-based UPS with much larger capacity, partially to bridge to renewable energy sources during grid disruptions.

### 2.3 Grid Interconnection Delays

The gating constraint on AI datacenter construction is increasingly the grid itself. Building a 1 GW datacenter requires the local utility to provide 1 GW of additional capacity, which often means new transmission lines, new substations, and (in some cases) new generation. The interconnection process can take years, and several major datacenter projects have been delayed by grid-side constraints rather than by construction or equipment availability.

Texas, Virginia, Arizona, and parts of the Midwest have been the most popular locations for AI datacenters, partly because they have relatively cheap and available power. But even in these locations, the rate of new datacenter construction is starting to outpace the rate of grid expansion, leading to interconnection queues that can stretch into the late 2020s.

### 2.4 Power Sourcing

The major hyperscalers have committed to powering their AI datacenters with renewable energy. Microsoft, Google, Amazon, and Meta all have ambitious renewable energy procurement programs, often through power purchase agreements with utility-scale solar and wind projects. The reality is more complicated: renewable energy is variable, AI workloads need constant power, and the storage capacity to bridge the gap is expensive.

The current state is a mix. Some AI datacenters are colocated with dedicated renewable generation. Most are connected to the grid and rely on the utility's mix of generation. Several hyperscalers have signed agreements with nuclear power plants — both existing reactors and proposed small modular reactors — to secure baseload renewable power for AI workloads. Microsoft's deal with Constellation Energy to restart a Three Mile Island reactor is the most prominent example.

The longer-term picture is uncertain. The combination of AI's growing power demand and the slower pace of clean generation expansion creates real tension. Some analysts predict that AI growth will force investment in natural gas generation despite the climate implications, simply because there is no other way to add capacity quickly enough. Others believe that the demand will catalyze accelerated renewable and nuclear investment.

## 3. Cooling

### 3.1 Why Air Cooling Fails

The fundamental physics of air cooling are unforgiving. To remove 120 kW from a single rack, an air-cooling system would need to move enormous volumes of air at very low temperatures, with airflow speeds that approach gale force. The fans required would consume substantial power themselves, and the noise would be deafening. In practice, air cooling is limited to perhaps 25-30 kW per rack with aggressive engineering, and that is well below modern AI rack densities.

Liquid cooling is the only practical alternative. Water can carry roughly 1000× more heat per unit volume than air at the same temperature differential, so even modest flow rates can remove large amounts of heat. Liquid cooling at the scale needed for AI datacenters is well-understood from supercomputing, but applying it to large-scale commercial datacenters is a new operational challenge.

### 3.2 Liquid Cooling Approaches

Three main approaches to liquid cooling are used in AI datacenters:

**Rear-Door Heat Exchangers (RDHX)** put a liquid-cooled radiator on the back of each rack. Hot air leaving the servers passes through the radiator and is cooled before returning to the room. This is a transitional technology that allows existing air-cooled servers to be deployed at higher densities. It can handle 30-50 kW per rack, which is enough for some AI workloads but inadequate for the highest-density configurations.

**Direct-to-Chip Liquid Cooling (DLC)** routes coolant directly to cold plates that sit on the hot components (GPUs, CPUs). The coolant absorbs heat from the chips and is pumped to a heat exchanger that dumps the heat to a secondary cooling loop. DLC can handle 100+ kW per rack and is the standard approach for NVL72-class racks. The Blackwell GB200 is designed from the outset for DLC.

**Immersion Cooling** submerges the entire server in a non-conductive fluid (typically a synthetic dielectric oil or 3M Novec, though Novec is being phased out for environmental reasons). The fluid absorbs heat from all components and is circulated through external heat exchangers. Immersion can handle the highest power densities (200+ kW per rack) and provides the lowest cooling power overhead. The downside is that maintenance is messier, and the supply chain for immersion-compatible servers is less developed.

For most new AI datacenters, DLC is the dominant choice. It provides enough cooling for current and near-future hardware, has a mature supply chain, and integrates with conventional rack designs. Immersion is gaining adoption in specific contexts (very high density, certain crypto and HPC workloads) but remains a smaller market.

### 3.3 Water Consumption

Liquid-cooled datacenters consume large amounts of water. The water is mostly recirculated in closed loops, but the heat eventually has to be rejected to the environment, typically through evaporative cooling towers that use large volumes of makeup water. A 1 GW datacenter can consume millions of gallons of water per day, comparable to a small city.

This water consumption is becoming controversial in water-stressed regions. Several proposed datacenter projects in Arizona, Nevada, and Texas have faced opposition from local communities concerned about water availability. Some hyperscalers have responded by investing in dry cooling (which uses much less water but more electricity), in waste-water reuse, or in designs that reject heat to the air rather than to evaporated water. The tradeoffs between water consumption, electricity consumption, and capital cost are real and increasingly visible.

### 3.4 Heat Reuse

A potential mitigation for the energy waste of cooling is to reuse the heat. The waste heat from a datacenter is not very high-grade (typically 30-50°C return temperatures from DLC, lower for air cooling), but it is suitable for some applications: district heating, greenhouses, swimming pool heating, certain industrial processes. Several European datacenters have implemented heat reuse systems that supply nearby buildings with waste heat from the datacenter.

In the US, heat reuse is less common because of the geographic isolation of most datacenters from heat-demanding facilities. The economic case for transporting low-grade heat over long distances is weak. As datacenters move to more populated areas to be near power and network infrastructure, heat reuse may become more viable.

## 4. Networking

### 4.1 Intra-Datacenter Networking

The networking requirements of AI training are extreme. A large training job might use thousands of GPUs across many racks, and the GPUs need to communicate at very high bandwidth and very low latency. The intra-datacenter network for AI is typically built around InfiniBand or RoCE (RDMA over Ethernet), with multi-tier fat-tree or dragonfly topologies.

The bandwidth per GPU is rising rapidly. Current generation deployments use 400 Gbps per GPU (NDR InfiniBand or 400 GbE). Next generation uses 800 Gbps (XDR or 800 GbE). The aggregate bandwidth of a large AI cluster reaches petabits per second, requiring custom optical interconnects, high-radix switches, and substantial cable plant.

The networking equipment is itself a major power and space consumer. A large AI cluster's switches can consume 5-10% of the total facility power, and the cabling alone can occupy entire dedicated rooms. Some new designs are exploring co-packaged optics, where the optical engines sit directly on the switch silicon, reducing the power and space overhead of the network fabric.

### 4.2 Inter-Datacenter Networking

For training runs that span multiple physical sites — increasingly common as single sites hit power and space limits — inter-datacenter networking becomes the bottleneck. The dedicated long-haul fiber required to connect distant datacenters at multi-terabit-per-second speeds is expensive, and the latency between sites limits the parallelism strategies that can be used effectively.

Most large training runs still happen within a single datacenter for this reason. The few exceptions (Google's TPU pod system, some experimental Microsoft training runs) use carefully-designed multi-site setups with dedicated fiber connecting the locations. The technology is improving but remains a constraint on the largest training jobs.

## 5. Building Design

### 5.1 Structural Considerations

AI datacenters have higher floor loads than conventional datacenters. A 120 kW liquid-cooled rack with all its supporting infrastructure can weigh 4-5 tons, concentrated in a small floor area. The building structure must support this load, which means thicker concrete slabs, more steel reinforcement, and (in some cases) reinforced foundations. Retrofitting an existing datacenter for AI workloads is often more expensive than building a new facility, because the structural changes are extensive.

The buildings themselves are also larger. A 1 GW AI datacenter is typically a campus of several buildings rather than a single building, because the power and cooling distribution is impractical at larger scales. The total footprint can exceed 100,000 square meters, comparable to a large factory or warehouse.

### 5.2 Modular and Container-Based Designs

Several hyperscalers and specialized datacenter companies are exploring modular designs, where the datacenter is assembled from prefabricated modules that contain all the necessary infrastructure (power, cooling, networking) for a specific number of racks. The modules are built in a factory, shipped to the site, and connected together. This reduces the construction time and allows the datacenter to be expanded incrementally.

The most prominent modular design is the OpenAI Stargate concept, which envisions massive AI datacenters built from prefabricated modules at sites with abundant power and cooling. The first Stargate sites are under construction in Texas, with capacity targets in the 1-5 GW range.

### 5.3 Site Selection

Where to put an AI datacenter is a multi-dimensional optimization. The factors include:

- **Power availability**: Cheap, abundant electricity, ideally with low carbon intensity
- **Water availability**: For cooling, where applicable
- **Network connectivity**: Proximity to fiber backbone networks
- **Climate**: Cool climates reduce cooling costs, dry climates reduce evaporation
- **Land cost**: Cheap land for the large footprint required
- **Tax incentives**: Many states and countries offer tax incentives for datacenter development
- **Skilled workforce**: For operations and maintenance
- **Latency to users**: Important for inference, less important for training

The result is that AI datacenters cluster in specific regions: the Pacific Northwest (cool climate, hydroelectric power), Texas (abundant land and power), Northern Virginia (existing datacenter ecosystem), Dublin (Ireland's datacenter hub), Northern Europe (cool climate, renewable power), and increasingly the Middle East (abundant solar power and specialized investment).

## 6. The Grid Impact

The aggregate impact of AI datacenters on electrical grids is becoming a major political and economic issue. Several US states have seen their projected electricity demand growth jump dramatically due to datacenter construction. Virginia's Dominion Energy projects 85% load growth over the next 15 years, almost entirely from datacenters. Texas's ERCOT is similarly forecasting massive datacenter-driven growth.

The grid has not historically been designed for this kind of demand growth. Adding gigawatts of new load in specific locations strains the existing transmission infrastructure and pushes utilities to invest in new generation. The pace of utility investment is much slower than the pace of datacenter construction, creating bottlenecks that affect both AI development and broader grid reliability.

Several mitigation strategies are being explored. Behind-the-meter generation (datacenters with their own gas turbines or solar arrays) reduces grid dependence. Demand response (datacenters that reduce load during grid stress) provides flexibility. Co-location with new generation (siting datacenters next to new wind, solar, or nuclear plants) avoids the transmission bottleneck. None of these fully solves the problem, but together they may slow the rate at which AI datacenters strain the grid.

## 7. The Geopolitical Dimension

AI datacenter construction has become a strategic asset. Countries are competing to attract AI datacenter investment, both for the economic benefits and for the implicit AI capability it represents. The US, China, EU, UK, Japan, South Korea, UAE, Saudi Arabia, and several other countries have explicit national strategies to build AI infrastructure.

The geographic concentration of AI compute matters for several reasons. Latency-sensitive workloads need to be near users. Sovereignty considerations push countries to host AI compute domestically. Export controls (especially the US restrictions on AI chips to China) shape where chips can be deployed. The result is a global map of AI capacity that is rapidly reshaping itself, with substantial investment in places that did not have major datacenter footprints just a few years ago.

## 8. Environmental Impact

The environmental cost of AI datacenters is one of the most contentious topics in the broader AI debate. The energy consumption is real, the carbon emissions (depending on power source) are real, and the water consumption is real. Whether these costs are offset by the benefits of AI is a question that depends on one's views about the value of AI capabilities themselves.

The optimistic view is that AI datacenters are accelerating the deployment of clean energy by creating massive demand that justifies new investment, that the per-task efficiency of AI systems is improving rapidly, and that the eventual benefits of AI (in fields like climate modeling, materials science, drug discovery) will far exceed the environmental costs. The pessimistic view is that AI is a luxury energy consumer that should be subordinated to climate priorities and that the rapid datacenter buildout is making climate goals harder to achieve.

The truth is probably somewhere in between. AI datacenters are real environmental costs, the costs are growing, and the mitigation strategies (efficiency, renewable power, heat reuse) are real but limited. The political and economic question of how to balance these costs against the benefits of AI is one that the next several years will have to answer.

## 9. Conclusion

AI datacenters represent a discrete shift in how computing infrastructure is built. The power densities, cooling requirements, networking topologies, and grid interactions are all different from conventional datacenters in ways that require new design approaches. The investment is enormous, the timeline is compressed, and the constraints (power availability, grid interconnection, water, environmental impact) are increasingly binding.

For practitioners deploying AI workloads, the relevance of datacenter design is mostly indirect. The capacity, performance, and cost of cloud AI services depend on the underlying datacenter infrastructure, but most users do not need to know the details. For organizations building their own AI infrastructure, the details matter enormously, and the choice of facility design can determine whether a project is viable.

The deeper significance is that AI infrastructure is becoming one of the largest physical investment categories in the global economy. The decisions being made about AI datacenter construction over the next several years will shape not just the AI industry but the broader patterns of energy use, grid investment, and physical infrastructure development. These decisions deserve more attention than they typically receive, because they will determine the practical limits of what AI can do — and what it costs the rest of the world to make it happen.
