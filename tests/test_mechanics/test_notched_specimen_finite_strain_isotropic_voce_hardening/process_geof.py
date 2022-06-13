respaht = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/meshes/SSNA03_QUA8R.geof"
respaht2 = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/meshes/SSNA03_QUA8R_2.geof"

nlines = []
with open(respaht, "r") as gfile:
    c = gfile.readlines()
    for line in c:
        nline = line.replace("  ", " ").replace("   ", " ").replace("    ", " ").replace("     ", " ").replace("      ", " ").replace("       ", " ").replace("        ", " ").replace(" ", "@")
        nline.replace("  ", " ").replace("   ", " ").replace("    ", " ").replace("     ", " ").replace("      ", " ").replace("       ", " ").replace("        ", " ")
        nlines.append(nline)
with open(respaht2, "w") as gfile2:
    for nnlije in nlines:
        # if nnlije[0] == " ":
        gg = nnlije
        print("there")
        print(gg)
        gfile2.write(gg)