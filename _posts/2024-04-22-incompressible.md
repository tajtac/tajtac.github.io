---
layout: post
title: incompressibility in hyperelasticity
date: 2024-04-22 12:56:00-0400
description: how incompressibility is enforced in hyperelasticity
tags: constitutive hyperelasticity
# categories: sample-posts
---

This is my first blog post on my website, and as such, the post is partly motivated by me forcing myself to start writing blog posts. But nevertheless, I think the material I discuss here is interesting and graduate students starting to work on constitutive modeling, and especially, hyperelasticity, will find it useful.

I mainly want to discuss the mechanism by which incompressibility is imposed in constitutive models of hyperelasticity. So let's start by defining hyperelasticity.

## What is hyperelasticity?
Formally, hyperelasticity is defined by means of an equation. We can try to describe hyperelasticity in words, but that would just consist of describing the equation in words, which is inefficient. If we denote the first Piola Kirchhoff stress with $$\mathbf{P}$$ and the deformation gradient with $$\mathbf{F}$$, then hyperelasticity posits the existence of a scalar-valued potential function known as the <i>strain energy density function</i> (SEDF) {% cite ogden1997non --collection external_references %}, denoted as $$\Psi(\mathbf{F})$$, such that

\begin{equation}
    \label{eq_def_hyperelasticity}
    \mathbf{P} = \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}} \, .
\end{equation}

The most important implication of this equation is that in hyperelasticity, the relationship between deformation and stress is <i>unchanging</i>. This is as opposed to other constitutive models such as viscoelasticity and continuum damage modeling, where $$\Psi$$ evolves with time, loading history or the state of damage in the material.

The deformation gradient $$\mathbf{F}$$ contains information both about stretching and rotations of a point in the material, however, $$\Psi$$ has to remain invariant of rigid body rotations. To this end, the SEDF is often expressed as a function of the <i>right Cauchy-Green deformation tensor</i>, $$\mathbf{C} = \mathbf{F}^\top \mathbf{F}$$. Using this form of the SEDF, the <i>Second Piola Kirchhoff Stress</i>, $$\mathbf{S}$$, can be obtained as

\begin{equation}
    \label{eq_S}
    \Psi = \Psi(\mathbf{C}), \quad \mathbf{S} = \frac12 \frac{\partial \Psi(\mathbf{C})}{\partial \mathbf{C}} \, .
\end{equation}

> Note: A lot of people assume that Eq. \eqref{eq_S} is equivalent to Eq. \eqref{eq_def_hyperelasticity}, or that it can be readily obtained from it by multiplying both sides of \eqref{eq_def_hyperelasticity} by $$\mathbf{F}^{-1}$$, but that is not the case. The fundamental equation of hyperelasticity is Eq. \eqref{eq_def_hyperelasticity}. Eq. \eqref{eq_S} can be derived from Eq. \eqref{eq_def_hyperelasticity} <i>under certain conditions</i>. See {% cite bonetNONLINEARCONTINUUMMECHANICS1997 --collection external_references %} for more details.


### Incompressible hyperelasticity
Eq. \eqref{eq_S} provides the general form of the relationship between $$\mathbf{C}$$ and $$\mathbf{S}$$ in hyperelasticity when $$\mathbf{C}$$ is <i>arbitrary</i>. However, under the condition of incompressibility, the elements of $$\mathbf{C}$$ are not arbitrary. In fact, when we impose incompressibility, we are confining $$\mathbf{C}$$ to a very special subspace of $$\mathbb{R}^3 \times \mathbb{R}^3$$ tensors such that the determinant of $$\mathbf{C}$$ is always 1. In this case the relationship between $$\mathbf{C}$$ and $$\mathbf{S}$$ has the following form {% cite bonetNONLINEARCONTINUUMMECHANICS1997 --collection external_references %}

\begin{equation}
    \label{eq_S_incomp}
    \mathbf{S} = 2 \frac{\partial \hat{\Psi}(\mathbf{C})}{\partial \mathbf{C}} + p J \mathbf{C}^{-1}
\end{equation}

where $$p$$ is a pressure Lagrange multiplier which can be understood as a hydrostatic stress to resist compression, $$J=\det \mathbf{F}$$ is the volume change and $$\hat{\Psi}$$ is a <i>distortional</i> energy function that depends on $$\mathbf{C}$$ only through the isochoric part of $$\mathbf{C}$$, i.e.,

\begin{equation}
    \hat{\Psi}(\mathbf{C}) = \Psi(\hat{\mathbf{C}}) \nonumber
\end{equation}
\begin{equation}
    \hat{\mathbf{C}} = (\det \mathbf{C})^{-1/3}\mathbf{C} \, . \nonumber
\end{equation}

But now, the question is, how to find it $$p$$? In some special cases, such as biaxial deformation of a thin membrane under plane-stress conditions, $$p$$ can be determined from boundary conditions. This is a widely used approach in mechanics of skin, for example.

### Nearly incompressible hyperelasticity
In general, finding $$p$$ analytically is not possible. Therefore, a <i>nearly incompressible</i> approach is followed in most practical applications {% cite holzapfel2002nonlinear --collection external_references %}. In nearly incompressible hyperelasticity, the SEDF is constructed by adding a volumetric term to the distortional energy, i.e.,

\begin{equation}
    \label{eq_S_comp_general}
    \Psi(\mathbf{C}) = \hat{\Psi}(\mathbf{C}) + \Psi_{\text{vol}}(J)
\end{equation}

$$\Psi_{\text{vol}}$$ is simply a term that penalizes volume changes in the material ($$J=1$$ corresponds to no volume change, while deviations from $$J=1$$ signify changing volume). This form of the SEDF is supposed to imitate incompressibility by requiring extremely high energies for the material to change volume (Note that $$\hat{\Psi}$$ does not depend on volume change). One simple form of $$\Psi_{\text{vol}}$$ is given by

\begin{equation}
    \label{eq_psivol}
    \Psi_{\text{vol}} = \frac{1}{2} K (J-1)^2 \, ,
\end{equation}

with $$K$$ a bulk modulus parameter. When this form of $$\Psi_{\text{vol}}$$ is used, the second Piola Kirchhoff stress can be obtained as

\begin{equation}
    \label{eq_S_comp}
    \mathbf{S} = 2\frac{\partial \hat{\Psi}(\mathbf{C})}{\partial \mathbf{C}} + K (J-1)J\mathbf{C}^{-1} .
\end{equation}

> Note 2: At this point, it is important to note that the mechanisms by which we arrived at (the 2nd terms of) Eqs. \eqref{eq_S_incomp} and \eqref{eq_S_comp} are completely different. The former is the result of rigorous mathematical manipulations on Eq. \eqref{eq_S} with the assumption that $$\mathbf{C}$$ is incompressible (i.e., $$\det \mathbf{C}=1$$), while the latter is simply a consequence of choosing the form \eqref{eq_psivol} for the volumetric term and assuming that $$\Psi$$ has the form \eqref{eq_S_comp_general}. And yet, notice the similarity between the two. If we set $$p=K(J-1)$$, the two forms become identical. If you are puzzled by this, know that I am too.

The only task that remains ahead before we can model the behavior of a hyperelastic material is to specify a suitable form of $$\hat{\Psi}$$. $$\hat{\Psi}$$ has to satisfy a number of mathematical and physical constraints to be admissible. First and foremost, $$\hat{\Psi}$$ has to be <i>objective</i>. Simply put, the principle of objectivity states that the stress in the material must be independent of the frame of reference. There are two widely used methods of satisfying this criterion: 1) using SEDFs that only depend on the principal (distortional) stretches, $$\hat{\lambda}_1, \hat{\lambda}_2, \hat{\lambda}_3$$ (square roots of the eigenvalues of $$\hat{\mathbf{C}}$$) {% cite lohr2022introduction ogden1997non --collection external_references %}, or 2) using SEDFs that only depend on the tensor invariants of $$\hat{\mathbf{C}}$$ {% cite ehret2007polyconvex --collection external_references %}. 

Defining suitable, and yet, flexible forms of $$\hat{\Psi}$$ has been an active area of research for quite a while. Soft materials display some very complex and varied set of behaviors and deriving one form of the SEDF that has the flexibility to explain all this complexity and variety in different materials is not easy. There are hundreds of proposed SEDF models in the literature (See the recent review paper by Dal et al. {% cite dalPerformanceIsotropicHyperelastic2021 --collection external_references %} for a list of 44 such models.), and more keep coming out every year.

My contribution to the solution of this problem was to use machine learning models to represent $$\hat{\Psi}$$ {% cite tacDatadrivenModelingMechanical2022 tacDatadrivenTissueMechanics2022 --collection external_references %}. 


## References
{% bibliography --cited %}