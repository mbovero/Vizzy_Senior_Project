from ik_callable import make_default_arm, ik_cmds_bounded
arm = make_default_arm()
cmds = ik_cmds_bounded(arm, (0.0, 0.0, 0.4), 0.0, radius_m=0.15, z_min=0.0, z_max=0.4)


print(f"{cmds['q1']:.2f} {cmds['q2']:.2f} {cmds['q3']:.2f} {int(round(cmds['q4']))}")

