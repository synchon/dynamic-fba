# -*- coding: utf-8 -*-

"""Contains function to perform lexicographic linear programming."""

from __future__ import absolute_import

from optlang.symbolics import Zero
import pandas as pd


def lexicographic_lp(model, rxn_list):
    """
    Perform Lexicographic Linear Programming.

    Parameters
    ----------
    model: cobra.Model
        The model to perform lexicographic LP on.
    rxn_list: list(cobra.Reaction.id, tuple(cobra.Reaction.id)
        The list containing the reactions to be considered as objectives.

    Returns
    -------
    pandas.Series
    """
    biomass_rxn_id, ex_rxn_ids = rxn_list

    with model:
        # LP feasibility
        obj_vars = []
        for met in model.metabolites:
            s = model.problem.Variable("s_" + met.id, lb=0)
            beta = model.problem.Variable("beta_" + met.id, lb=0)
            s_equal_beta_const = model.problem.Constraint(
                s - beta,
                name="s_equal_beta_" + met.id, ub=0.0, lb=0.0)
            model.add_cons_vars([s, beta, s_equal_beta_const])
            model.constraints[met.id].set_linear_coefficients({s: 1.0,
                                                               beta: -1.0})
            obj_vars.append(s)
        model.objective = model.problem.Objective(Zero,
                                                  sloppy=True,
                                                  direction="min")
        model.objective.set_linear_coefficients({v: 1.0 for v in obj_vars})
        model.objective_direction = "min"
        sol_feasibility = model.slim_optimize()
        feasibility_constraint = model.problem.Constraint(
            model.objective.expression,
            name="fixed_feasibility", ub=sol_feasibility, lb=sol_feasibility)
        model.add_cons_vars([feasibility_constraint])

        # Biomass
        model.objective = model.reactions.get_by_id(biomass_rxn_id)
        model.objective_direction = "max"
        sol_biomass = model.slim_optimize()
        biomass_constraint = model.problem.Constraint(
            model.objective.expression,
            name="fixed_biomass", ub=sol_biomass, lb=sol_biomass)
        model.add_cons_vars([biomass_constraint])

        # Exchanges
        for rxn_id in ex_rxn_ids:
            model.objective = model.reactions.get_by_id(rxn_id)
            model.objective_direction = "min"
            sol = model.slim_optimize()
            exchange_constraint = model.problem.Constraint(
                model.objective.expression,
                name="fixed_" + rxn_id, ub=sol, lb=sol)
            model.add_cons_vars([exchange_constraint])

        # Indexing list
        loc_list = [biomass_rxn_id]
        loc_list.extend(list(ex_rxn_ids))

        sol = model.optimize()
        return pd.Series(sol.fluxes.loc[loc_list])