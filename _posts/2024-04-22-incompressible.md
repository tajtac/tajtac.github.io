---
layout: post
title: incompressibility in hyperelasticity
date: 2024-04-22 12:56:00-0400
description: how incompressibility is enforced in hyperelasticity
tags: constitutive hyperelasticity
# categories: sample-posts
---
## Hyperelasticity
Hyperelasticity is formally defined in terms of an energy potential that allows us to relate the stress at a point in the material to the deformation at that point. If we denote the first Piola Kirchhoff stress with $$\mathbf{P}$$ and the deformation gradient with $$\mathbf{F}$$, then hyperelasticity posits the existence of a scalar-valued potential function known as the \emph{strain energy density function} (SEDF) {% cite ogden1997non %}, denoted as $$\Psi(\mathbf{F})$$, such that

\begin{equation}
    \label{eq_def_hyperelasticity}
    \mathbf{P} = \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}} \, .
\end{equation}

The most important implication of this equation is that in hyperelasticity, the relationship between deformation and stress is \emph{unchanging}. This is as opposed to other constitutive models such as viscoelasticity and continuum damage modeling, where $$\Psi$$ evolves with time, loading history or the state of damage in the material (as we will see later on).

The deformation gradient $$\mathbf{F}$$ contains information both about stretching and rotations of a point in the material, however, $$\Psi$$ has to remain invariant of rigid body rotations. To this end, the SEDF is often expressed as a function of the \emph{right Cauchy-Green deformation tensor}, $$\mathbf{C} = \mathbf{F}^\top \mathbf{F}$$. Using this form of the SEDF, the \emph{Second Piola Kirchhoff Stress}, $$\mathbf{S}$$, can be obtained as

\begin{equation}
    \label{eq_S}
    \Psi = \Psi(\mathbf{C}), \quad \mathbf{S} = \frac12 \frac{\partial \Psi(\mathbf{C})}{\partial \mathbf{C}} \, .
\end{equation}


### Incompressible hyperelasticity
Eq. \eqref{eq_S} provides the most general form of the relationship between $$\mathbf{C}$$ and $$\mathbf{S}$$ in hyperelasticity when $$\mathbf{C}$$ is arbitrary. However, under the condition of incompressibility, the elements of $$\mathbf{C}$$ are not arbitrary. In this case the relationship between $$\mathbf{C}$$ and $$\mathbf{S}$$ has the following form \cite{bonetNONLINEARCONTINUUMMECHANICS1997}

\begin{equation}
    \label{eq_S_incomp}
    \mathbf{S} = 2 \frac{\partial \hat{\Psi}(\mathbf{C})}{\partial \mathbf{C}} + p J \mathbf{C}^{-1}
\end{equation}

where $$p$$ is a pressure Lagrange multiplier which can be understood as a hydrostatic stress to resist compression, $$J=\det \mathbf{F}$$ is the volume change and $$\hat{\Psi}$$ is a \emph{distortional} energy function that depends on $$\mathbf{C}$$ only through the isochoric part of $$\mathbf{C}$$, i.e.,

\begin{equation*}
    \hat{\Psi}(\mathbf{C}) &= \Psi(\hat{\mathbf{C}}) \\
    \hat{\mathbf{C}} &= (\det \mathbf{C})^{-1/3}\mathbf{C} \, .
\end{equation*}

Using the relationship in Eq. \eqref{eq_S_incomp} requires determining the pressure $$p$$. In some special cases, such as biaxial deformation of a thin membrane under plane-stress conditions, $$p$$ can be determined from boundary conditions. However, this is not possible in general. Therefore, a \emph{nearly incompressible} approach is followed in most practical applications \cite{holzapfel2002nonlinear}. In nearly incompressible hyperelasticity of soft tissue, the SEDF is constructed by adding a volumetric term to the distortional energy, i.e.,

\begin{equation*}
    \Psi(\mathbf{C}) = \hat{\Psi}(\mathbf{C}) + \Psi_{\text{vol}}(J)
\end{equation*}

$$\Psi_{\text{vol}}$$ is simply a term that penalizes volume changes in the material ($$J=1$$ corresponds to no volume change, while deviations from $$J=1$$ signify changing volume). The simplest form of $$\Psi_{\text{vol}}$$ is given by

\begin{equation}
    \label{eq_psivol}
    \Psi_{\text{vol}} = \frac{1}{2} K (J-1)^2 \, ,
\end{equation}

with $$K$$ a bulk modulus parameter. When this form of $$\Psi_{\text{vol}}$$ is used, the second Piola Kirchhoff stress can be obtained as

\begin{equation}
    \label{eq_S_comp}
    \mathbf{S} = 2\frac{\partial \hat{\Psi}(\mathbf{C})}{\partial \mathbf{C}} + K (J-1)J\mathbf{C}^{-1} .
\end{equation}

The only task that remains ahead before we can model the behavior of a hyperelastic material is to specify a suitable form of $$\hat{\Psi}$$. $$\hat{\Psi}$$ has to satisfy a number of mathematical and physical constraints to be admissible. First and foremost, $$\hat{\Psi}$$ has to be \emph{objective}. Simply put, the principle of objectivity states that the stress in the material must be independent of the frame of reference. There are two widely used methods of satisfying this criterion: 1) using SEDFs that only depend on the principal (distortional) stretches, $$\hat{\lambda}_1, \hat{\lambda}_2, \hat{\lambda}_3$$ (square roots of the eigenvalues of $$\hat{\mathbf{C}}$$) \cite{lohr2022introduction,ogden1997non}, or 2) using SEDFs that only depend on the tensor invariants of $$\hat{\mathbf{C}}$$ \cite{ehret2007polyconvex}. 

