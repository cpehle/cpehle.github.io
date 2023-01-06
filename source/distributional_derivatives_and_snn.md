---
title: Distributional Derivatives and Spiking Neural Networks
author:
- name: Christian Pehle
  email: christian.pehle@kip.uni-heidelberg.de
fontfamily: times
csl: american-physics-society.csl
link-citations: true
type: Draft
---

Spiking neural networks exchange "spikes" in order to perform computations. Spikes occur at certain points in time and can be physically interpreted as an abstraction of a physical process that occurs much faster than the dynamics under consideration. To model sequences of spikes sums of delta distributions are commonly used:
$$
s(t) = \sum_i \delta(t - t_i)
$$
Physicists routinely abuse notation and treat distributions as if they were ordinary functions, we will do so in this exposition for the most part as well, with the understanding that with some more effort the arguments presented here could be made rigorous.

To define the derivative of a distribution one uses a trick perfected by mathematicians: Simply define what you want to be true, to be the definition. There is a pairing between distributions and smooth compactly supported functions, which physicists simply write as the integral
$$
\langle \eta, f \rangle = \int_{-\infty}^{\infty}  \eta f \mathrm{dt}
$$
the derivative $\eta'$ of a distribution is defined as
$$
\langle \eta', f \rangle = -\langle \eta, f' \rangle = -\int_{-\infty}^{\infty}  \eta f' \mathrm{dt}
$$

We can then use any sequence of functions, which is at least once differentiable almost everywhere as an approximation. As a simple example consider the sequence of triangle function $\Lambda_\epsilon$
$$
\delta(t) = \lim_{\epsilon \to 0} \frac{1}{\epsilon} \Lambda(t/\epsilon).
$$
Its derivative is given by
$$
\Lambda'_\epsilon = \dfrac{\Pi\left(\frac{t}{\epsilon} +\frac{\epsilon}{2}\right)-\Pi\left(\frac{t}{\epsilon} -\frac{\epsilon}{2}\right)}{\epsilon^2}.
$$
Now the second identity to keep in mind is that if $g(t)$ is a differentiable function of $t$, then
$$
\delta(g(t)) = \sum_{i} \frac{1}{\lvert g'(t) \rvert}\delta(t - t_i)
$$
and therefore
$$
\delta(g(t)) \lvert g'(t) \rvert = \sum_{i} \delta(t - t_i)
$$

If we now consider $N$ neurons and dynamical equations

$$
\begin{align}
\dot{V} &= f(V,I) \\
0       &= I - W_r s \\
\tau_s \dot{s}_k &= -s_k+ \delta(v_k - v_\mathrm{th}) \lvert \dot{v}_k(t) \rvert
\end{align}
$$

Consider a loss
$$
L = \int_0^T l(V,I,s,p) \mathrm{dt}
$$
and 
$$
\mathcal{L} = L + \langle \lambda_V, \dot{V} - f(V,I)
\rangle + \langle \lambda_I, I - W_r s\rangle + \langle \lambda_s, -s + \delta(v - v_\mathrm{th}) \lvert \dot{v}_k(t) \rvert \rangle
$$
