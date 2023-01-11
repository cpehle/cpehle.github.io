---
title: Research Statement
author:
- name: Christian Pehle
  email: christian.pehle@kip.uni-heidelberg.de
fontfamily: times
csl: american-physics-society.csl
link-citations: true
---

## Overview {#overview .unnumbered}

My research has focussed on Machine Learning in the context of
Brain-Inspired or Neuromorphic Computing, where I have made hardware
[@pehle2022brainscales; @schreiber2020closed; @aamir2018accelerated],
software [@norse2021], algorithmic [@wunderlich2021event], as well as
conceptual contributions
[@wunderlich2021event; @bohnstingl2019neuromorphic; @pehle2020neuromorphic].

## Event-based backpropagation {#event-based-backpropagation .unnumbered}

I have derived, in collaboration with Timo Wunderlich, the analogue of
the backpropagation algorithm for continuous time spiking neural
networks [@wunderlich2021event] -- EventProp. It computes exact
parameter gradients in networks of spiking neurons with no restrictions
on network topology or loss function. Previous work had either found
solutions for particular cases or considered finding exact parameter
gradients impossible due to the discontinuous nature of spike
transitions. Notably, the algorithm can be efficiently implemented using
an event-based simulation algorithm and on digital neuromorphic
hardware, requiring only temporally sparse communication during the
backward phase. Furthermore, the memory requirements are proportional to
the number of communicated spikes resulting in order of magnitude
improvement relative to previous approaches. These properties also
strongly suggest an energy-efficient implementation in next-generation
analog neuromorphic hardware is possible.

In ongoing work, we have demonstrated that the algorithm can estimate
parameter gradients by observing only spikes from an analog Neuromorphic
Hardware emulating a known spiking neural network
[@pehle2022eventbased]. We match a previous approach
[@cramer2022surrogate] in performance (on tasks we have evaluated so
far) while not requiring densely sampled membrane-voltage traces. Such
dense observation of the system state is far less information efficient
and would be prohibitive for large-scale systems.

While EventProp cannot be considered biologically plausible, I believe
it to be one key enabling "technology" for NeuroAI [@zador2022toward] --
analogous to how the backpropagation algorithm was foundational to the
current rapid progress in Machine Learning -- once it is combined with
further insights from neuroscience, such as connectivity and neuron
dynamics, as well as the large scale electro-physiological and
connectivity data.

There also exists an analog of real-time recurrent learning (RTRL)
[@williams1989experimental] for spiking neural networks and
corresponding approximate truncations, which I derived in my thesis
[@pehle2021adjoint], but have yet to evaluate experimentally.

## Correlations in Spiking Neural Networks {#correlations-in-spiking-neural-networks .unnumbered}

I've investigated, in collaboration with Christof Wetterich, how
correlations -- fundamental both to quantum mechanics and potentially
computation in biological systems -- can be learned in networks of
spiking neurons [@pehle2020neuromorphic]. One main contribution of this
work was a first demonstration that correlations can approximate
specific low-dimensional quantum density matrices in networks of spiking
neurons. Another contribution is the demonstration that it is possible
to implement neural sampling using an end-to-end learning approach
without any prior assumptions on the underlying equilibrium probability
distribution. Further research in this direction could yield a practical
way to approximate a restricted set of density matrices, for example,
ground states of particular quantum spin systems, or to perform quantum
state tomography. It also could inform theoretical tools for studying
correlations in spiking neural networks.

## Hardware Design and Plasticity Experiments {#hardware-design-and-plasticity-experiments .unnumbered}

I was part of the design team of an analog Neuromorphic Processor
[@pehle2022brainscales] -- BrainScaleS-2. In particular, I was
responsible for the scaling up and verification of the "plasticity
processing unit" (an embedded processor with a
single-instruction-multiple data unit and parallel access to analog
system observables). The purpose of this processor is to enable "hybrid
plasticity," that is, the flexible implementation of bio-inspired
learning rules that can directly interact with the analog emulation of
neuron dynamics [@friedmann2016demonstrating]. Since the analog
components have time constants that are approximately $10^3$ faster than
the time constants of biological neuron voltage dynamics, it enables the
rapid evaluation of plasticity rules over long biological timescales, as
well as the use of evolutionary algorithms. I evaluated and designed
plasticity experiments and extended the digital implementation based on
user requirements. I interacted with both computational and experimental
neuroscientists to determine which current computational approaches to
bio-plausible learning and plasticity could be realized. My main
contributions were: To suggest the implementation of meta-plasticity
(implemented by executing a small ANN on the plasticity processing unit,
whose weights were optimized using evolutionary strategies) and
contributing to a demonstration of Learning-to-Learn on Neuromorphic
Hardware [@bohnstingl2019neuromorphic]. To implement a memory interface,
which enabled hardware-in-the-loop learning based on membrane trace
measurements [@cramer2022surrogate]. To implement spike routing and I/O
for a prototype system, which enabled closed-loop experiments
[@schreiber2020closed] and a first demonstration of R-STDP reinforcement
learning [@wunderlich2019demonstrating].

## Convenient SNN training compatible with Deep Learning {#convenient-snn-training-compatible-with-deep-learning .unnumbered}

I created and co-develop one of the first software libraries, which
allows non-experts to train and simulate spiking neural networks in a
way that is readily interoperable with common concepts from deep
learning -- Norse [@norse2021]. Prior work had either used programming
abstractions taken from neuron simulators, like populations and
projections or was tightly coupled to implementation choices of a
specific publication, with little chance of reuse. Norse has now been
adopted by several groups and is in active use.

---------------- 