# -*- coding: utf-8 -*-

"""Contains function to perform dynamic FBA."""

from __future__ import absolute_import

from dynamicfba.lexicographic import lexicographic_lp

from scipy.integrate import solve_ivp


def dynamic_fba(model_and_rxns, kinetic_eq, time_span, initial_conc_list):
    """
    Perform dynamic FBA to simulate batch growth of an organism.

    Dynamic FBA finds application in industrial fermentation process
    and modeling community growth of microorganisms, among others. It
    is an important tool which enhances FBA by adding reaction and metabolite
    dynamics. It can be performed mainly using 3 techniques:
        (1) Static Optimization Approach (SOA),
        (2) Dynamic Optimization Approach, and
        (3) Direct Approach.
    This algorithm uses the Direct Approach which circumvents the
    inaccuracy of SOA and the Non-Linear Programming (NLP) complexity
    found in DOA. This approach employs Lexicographic Linear Programming
    to provide unique flux distribution and also solves the LP
    feasibility problem and the bound problem when integrating.

    Parameters
    ----------
    model_and_rxns: dict(model: cobra.Model, biomass: cobra.Reaction.id,
                         exchanges: tuple(cobra.Reaction.id))
        The model(s) to perform dynamic FBA on and the metabolites to track.
    kinetic_eq: func()
        The kinetic equation to be used for dynamic simulation.
    Returns
    -------
    pandas.DataFrame

    References
    ----------
    .. [1] Gomez et al.: DFBAlab: a fast and reliable MATLAB code for dynamic
           flux balance analysis. BMC Bioinformatics 2014 15:409.
    """
    model = model_and_rxns["model"]
    biomass_rxn_id = model_and_rxns["biomass"]
    exchanges_rxn_id = model_and_rxns["exchanges"]
    rxn_list = [biomass_rxn_id]
    rxn_list.append(exchanges_rxn_id)
    # 1. LLP
    fluxes = lexicographic_lp(model, rxn_list)
    # 2. ODE integration
    # a. modify "y" so that if fits into the kinetic equation
    # of the form "kinetic_eq(t, y)"
    # b. set y0 to the initial concentrations of the metabolites
    solution = solve_ivp(kinetic_eq, time_span, initial_conc_list,
                         method='LSODA')
    return solution
