# Details

Date : 2022-06-13 20:02:42

Directory /home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/h2o

Total : 85 files,  40597 codes, 5508 comments, 842 blanks, all 46947 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [h2o/__init__.py](/h2o/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/__init__.py](/h2o/fem/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/basis/__init__.py](/h2o/fem/basis/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/basis/bases/__init__.py](/h2o/fem/basis/bases/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/basis/bases/monomial.py](/h2o/fem/basis/bases/monomial.py) | Python | 104 | 82 | 14 | 200 |
| [h2o/fem/basis/basis.py](/h2o/fem/basis/basis.py) | Python | 20 | 7 | 5 | 32 |
| [h2o/fem/element/__init__.py](/h2o/fem/element/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/element/damage_element.py](/h2o/fem/element/damage_element.py) | Python | 86 | 53 | 9 | 148 |
| [h2o/fem/element/displacement_element.py](/h2o/fem/element/displacement_element.py) | Python | 86 | 48 | 8 | 142 |
| [h2o/fem/element/element.py](/h2o/fem/element/element.py) | Python | 164 | 62 | 12 | 238 |
| [h2o/fem/element/finite_element.py](/h2o/fem/element/finite_element.py) | Python | 46 | 19 | 3 | 68 |
| [h2o/fem/element/operators/__init__.py](/h2o/fem/element/operators/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/fem/element/operators/gradient.py](/h2o/fem/element/operators/gradient.py) | Python | 387 | 13 | 14 | 414 |
| [h2o/fem/element/operators/identity.py](/h2o/fem/element/operators/identity.py) | Python | 33 | 2 | 4 | 39 |
| [h2o/fem/element/operators/stabilization.py](/h2o/fem/element/operators/stabilization.py) | Python | 446 | 82 | 8 | 536 |
| [h2o/field/__init__.py](/h2o/field/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/field/field.py](/h2o/field/field.py) | Python | 64 | 58 | 4 | 126 |
| [h2o/field/fields/__init__.py](/h2o/field/fields/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/field/fields/field_displacement.py](/h2o/field/fields/field_displacement.py) | Python | 128 | 28 | 12 | 168 |
| [h2o/field/fields/field_scalar.py](/h2o/field/fields/field_scalar.py) | Python | 19 | 4 | 3 | 26 |
| [h2o/geometry/__init__.py](/h2o/geometry/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/geometry/geometry.py](/h2o/geometry/geometry.py) | Python | 83 | 44 | 11 | 138 |
| [h2o/geometry/quadratures/__init__.py](/h2o/geometry/quadratures/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/geometry/quadratures/gauss/__init__.py](/h2o/geometry/quadratures/gauss/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/geometry/quadratures/gauss/gauss_hexahedron.py](/h2o/geometry/quadratures/gauss/gauss_hexahedron.py) | Python | 19,162 | 26 | 8 | 19,196 |
| [h2o/geometry/quadratures/gauss/gauss_quadrangle.py](/h2o/geometry/quadratures/gauss/gauss_quadrangle.py) | Python | 624 | 18 | 6 | 648 |
| [h2o/geometry/quadratures/gauss/gauss_quadrangle2.py](/h2o/geometry/quadratures/gauss/gauss_quadrangle2.py) | Python | 1,719 | 18 | 4 | 1,741 |
| [h2o/geometry/quadratures/gauss/gauss_segment.py](/h2o/geometry/quadratures/gauss/gauss_segment.py) | Python | 264 | 128 | 5 | 397 |
| [h2o/geometry/quadratures/gauss/gauss_tetrahedron.py](/h2o/geometry/quadratures/gauss/gauss_tetrahedron.py) | Python | 1,970 | 18 | 4 | 1,992 |
| [h2o/geometry/quadratures/gauss/gauss_triangle.py](/h2o/geometry/quadratures/gauss/gauss_triangle.py) | Python | 592 | 190 | 5 | 787 |
| [h2o/geometry/quadratures/quad2/__init__.py](/h2o/geometry/quadratures/quad2/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/geometry/quadratures/quad2/tri.py](/h2o/geometry/quadratures/quad2/tri.py) | Python | 568 | 18 | 4 | 590 |
| [h2o/geometry/quadratures/quadrature.py](/h2o/geometry/quadratures/quadrature.py) | Python | 6 | 0 | 3 | 9 |
| [h2o/geometry/shape.py](/h2o/geometry/shape.py) | Python | 247 | 133 | 31 | 411 |
| [h2o/geometry/shapes/__init__.py](/h2o/geometry/shapes/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/geometry/shapes/shape_hexahedron.py](/h2o/geometry/shapes/shape_hexahedron.py) | Python | 80 | 105 | 21 | 206 |
| [h2o/geometry/shapes/shape_polygon.py](/h2o/geometry/shapes/shape_polygon.py) | Python | 129 | 116 | 27 | 272 |
| [h2o/geometry/shapes/shape_polyhedron.py](/h2o/geometry/shapes/shape_polyhedron.py) | Python | 118 | 127 | 25 | 270 |
| [h2o/geometry/shapes/shape_quadrangle.py](/h2o/geometry/shapes/shape_quadrangle.py) | Python | 102 | 82 | 21 | 205 |
| [h2o/geometry/shapes/shape_quadrangle_diskpp.py](/h2o/geometry/shapes/shape_quadrangle_diskpp.py) | Python | 126 | 118 | 27 | 271 |
| [h2o/geometry/shapes/shape_segment.py](/h2o/geometry/shapes/shape_segment.py) | Python | 77 | 75 | 19 | 171 |
| [h2o/geometry/shapes/shape_tetrahedron.py](/h2o/geometry/shapes/shape_tetrahedron.py) | Python | 78 | 99 | 21 | 198 |
| [h2o/geometry/shapes/shape_triangle.py](/h2o/geometry/shapes/shape_triangle.py) | Python | 101 | 98 | 21 | 220 |
| [h2o/h2o.py](/h2o/h2o.py) | Python | 93 | 11 | 36 | 140 |
| [h2o/mesh/__init__.py](/h2o/mesh/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/mesh/element_description.py](/h2o/mesh/element_description.py) | Python | 71 | 70 | 15 | 156 |
| [h2o/mesh/gmsh/__init__.py](/h2o/mesh/gmsh/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/mesh/gmsh/data.py](/h2o/mesh/gmsh/data.py) | Python | 161 | 78 | 7 | 246 |
| [h2o/mesh/gmsh/expprt.py](/h2o/mesh/gmsh/expprt.py) | Python | 317 | 61 | 23 | 401 |
| [h2o/mesh/mesh.py](/h2o/mesh/mesh.py) | Python | 158 | 67 | 11 | 236 |
| [h2o/mesh/parsers/__init__.py](/h2o/mesh/parsers/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/mesh/parsers/geof.py](/h2o/mesh/parsers/geof.py) | Python | 222 | 97 | 25 | 344 |
| [h2o/problem/__init__.py](/h2o/problem/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/problem/boundary_condition.py](/h2o/problem/boundary_condition.py) | Python | 24 | 14 | 5 | 43 |
| [h2o/problem/coupled_problem.py](/h2o/problem/coupled_problem.py) | Python | 184 | 21 | 11 | 216 |
| [h2o/problem/coupled_problem2.py](/h2o/problem/coupled_problem2.py) | Python | 317 | 76 | 11 | 404 |
| [h2o/problem/damage_finite_element_field.py](/h2o/problem/damage_finite_element_field.py) | Python | 778 | 240 | 33 | 1,051 |
| [h2o/problem/displacement_finite_element_field.py](/h2o/problem/displacement_finite_element_field.py) | Python | 926 | 218 | 41 | 1,185 |
| [h2o/problem/finite_element_field.py](/h2o/problem/finite_element_field.py) | Python | 870 | 200 | 39 | 1,109 |
| [h2o/problem/load.py](/h2o/problem/load.py) | Python | 8 | 6 | 4 | 18 |
| [h2o/problem/material.py](/h2o/problem/material.py) | Python | 65 | 28 | 8 | 101 |
| [h2o/problem/output.py](/h2o/problem/output.py) | Python | 50 | 0 | 3 | 53 |
| [h2o/problem/problem.py](/h2o/problem/problem.py) | Python | 707 | 160 | 33 | 900 |
| [h2o/problem/resolution/__init__.py](/h2o/problem/resolution/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [h2o/problem/resolution/cond_backup.py](/h2o/problem/resolution/cond_backup.py) | Python | 427 | 138 | 4 | 569 |
| [h2o/problem/resolution/cond_backup2.py](/h2o/problem/resolution/cond_backup2.py) | Python | 349 | 80 | 5 | 434 |
| [h2o/problem/resolution/condensation.py](/h2o/problem/resolution/condensation.py) | Python | 349 | 279 | 5 | 633 |
| [h2o/problem/resolution/exact.py](/h2o/problem/resolution/exact.py) | Python | 309 | 162 | 2 | 473 |
| [h2o/problem/resolution/exact_backup2.py](/h2o/problem/resolution/exact_backup2.py) | Python | 347 | 103 | 5 | 455 |
| [h2o/problem/resolution/generic_backup.py](/h2o/problem/resolution/generic_backup.py) | Python | 590 | 154 | 10 | 754 |
| [h2o/problem/resolution/local_equilibrium.py](/h2o/problem/resolution/local_equilibrium.py) | Python | 344 | 62 | 4 | 410 |
| [h2o/problem/resolution/local_equilibrium2.py](/h2o/problem/resolution/local_equilibrium2.py) | Python | 463 | 136 | 7 | 606 |
| [h2o/problem/resolution/local_equilibrium_3.py](/h2o/problem/resolution/local_equilibrium_3.py) | Python | 370 | 96 | 4 | 470 |
| [h2o/problem/resolution/local_equilibrium_4.py](/h2o/problem/resolution/local_equilibrium_4.py) | Python | 492 | 147 | 8 | 647 |
| [h2o/problem/resolution/newton_solver.py](/h2o/problem/resolution/newton_solver.py) | Python | 195 | 44 | 15 | 254 |
| [h2o/problem/resolution/phase_field.py](/h2o/problem/resolution/phase_field.py) | Python | 497 | 56 | 10 | 563 |
| [h2o/problem/resolution/resolution.py](/h2o/problem/resolution/resolution.py) | Python | 0 | 154 | 1 | 155 |
| [h2o/problem/resolution/solve_element_inner_equilibrium.py](/h2o/problem/resolution/solve_element_inner_equilibrium.py) | Python | 68 | 7 | 5 | 80 |
| [h2o/problem/resolution/solve_generic.py](/h2o/problem/resolution/solve_generic.py) | Python | 610 | 156 | 10 | 776 |
| [h2o/problem/resolution/solve_linear_system.py](/h2o/problem/resolution/solve_linear_system.py) | Python | 490 | 140 | 13 | 643 |
| [h2o/problem/resolution/solve_static_condensation_thermo.py](/h2o/problem/resolution/solve_static_condensation_thermo.py) | Python | 532 | 164 | 13 | 709 |
| [h2o/problem/resolution/solve_static_condensation_thermo_axi.py](/h2o/problem/resolution/solve_static_condensation_thermo_axi.py) | Python | 561 | 95 | 11 | 667 |
| [h2o/problem/resolution/static_condensation.py](/h2o/problem/resolution/static_condensation.py) | Python | 497 | 56 | 10 | 563 |
| [h2o/problem/resolution/static_condensation_axisymmetric.py](/h2o/problem/resolution/static_condensation_axisymmetric.py) | Python | 510 | 61 | 10 | 581 |
| [h2o/problem/resolution/traction_force.py](/h2o/problem/resolution/traction_force.py) | Python | 17 | 0 | 8 | 25 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)