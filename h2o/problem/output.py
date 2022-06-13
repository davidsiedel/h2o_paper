from h2o.problem.problem import Problem
from h2o.problem.material import Material

import os

def create_output_txt(output_file_path: str, problem: Problem, material: Material):
    with open(output_file_path, "w") as outfile:
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** TIME STEPS :\n")
        for i, ts in enumerate(problem.time_steps):
            if i != len(problem.time_steps) - 1:
                outfile.write("{:.6E}, ".format(ts))
            else:
                outfile.write("{:.6E}\n".format(ts))
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** MAXIMUM NUMBER OF ITERATIONS PER STEP :\n")
        outfile.write("{}\n".format(problem.number_of_iterations))
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** FIELD :\n")
        outfile.write("DERIVATION TYPE : {}\n".format(problem.field.derivation_type.name))
        outfile.write("FILED TYPE : {}\n".format(problem.field.field_type.name))
        outfile.write("GRAD TYPE : {}\n".format(problem.field.grad_type.name))
        outfile.write("FLUX TYPE : {}\n".format(problem.field.flux_type.name))
        outfile.write("FIELD DIMENSION : {}\n".format(problem.field.field_dimension))
        outfile.write("EUCLIDEAN DIMENSION : {}\n".format(problem.field.euclidean_dimension))
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** FINITE ELEMENT :\n")
        outfile.write("FINITE ELEMENT TYPE : {}\n".format(problem.finite_element.element_type.name))
        outfile.write("CONSTRUCTION INTEGRATION ORDER : {}\n".format(problem.finite_element.construction_integration_order))
        outfile.write("COMPUTATION INTEGRATION ORDER : {}\n".format(problem.finite_element.computation_integration_order))
        outfile.write("FACE BASIS K ORDER : {}\n".format(problem.finite_element.face_basis_k.polynomial_order))
        outfile.write("CELL BASIS L ORDER : {}\n".format(problem.finite_element.cell_basis_l.polynomial_order))
        outfile.write("CELL BASIS K ORDER : {}\n".format(problem.finite_element.cell_basis_k.polynomial_order))
        outfile.write("CELL BASIS R ORDER : {}\n".format(problem.finite_element.cell_basis_r.polynomial_order))
        outfile.write("FACE BASIS K DIMENSION : {}\n".format(problem.finite_element.face_basis_k.dimension))
        outfile.write("CELL BASIS L DIMENSION : {}\n".format(problem.finite_element.cell_basis_l.dimension))
        outfile.write("CELL BASIS K DIMENSION : {}\n".format(problem.finite_element.cell_basis_k.dimension))
        outfile.write("CELL BASIS R DIMENSION : {}\n".format(problem.finite_element.cell_basis_r.dimension))
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** BOUNDARY CONDITIONS :\n")
        for i, bc in enumerate(problem.boundary_conditions):
            outfile.write("++++++ BOUNDARY : {}\n".format(bc.boundary_name))
            outfile.write("BOUNDARY TYPE : {}\n".format(bc.boundary_type.name))
            outfile.write("DIRECTION : {}\n".format(bc.direction))
        outfile.write("----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** MATERIAL :\n")
        outfile.write("STABILIZATION PARAMETER : {}\n".format(material.stabilization_parameter))
        outfile.write("BEHAVIOUR NAME : {}\n".format(material.behaviour_name))
        outfile.write("BEHAVIOUR INTEGRATION TYPE : {}\n".format(material.integration_type))
        outfile.write("NUMBER OF QUADRATURE POINTS : {}\n".format(material.nq))
        outfile.write("LAGRANGE PARAMETER : {}\n".format(material.lagrange_parameter))
        outfile.write("TEMPERATURE (K) : {}\n".format(material.temperature))
