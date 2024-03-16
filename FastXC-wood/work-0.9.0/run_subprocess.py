from fastxc.fastxc import FastXC
import subprocess

ini_file = '/home/woodwood/hpc/Resources/FastXC-wood/work-0.9.0/config_hinet.ini'

job = FastXC(ini_file)

# generate filter
job.generate_filter()

# sac2spec
job.generate_sac2spec_list_dir()

cmds_sac2spec_list = job.generate_sac2spec_cmd()

job.deploy_sac2spec_cmd()

# loop excute cmds_sac2spec_list
for cmd_sac2spec in cmds_sac2spec_list:
    print(cmd_sac2spec)
    subprocess.run(cmd_sac2spec.split(), check=True)

# bash_command = 


# # xc
# job.generate_xc_list_dir()

# job.generate_xc_cmd()

# job.deploy_xc_cmd()

# # stacking
# job.generate_stack_list_dir()

# job.generate_stack_cmd()

# job.deploy_stack_cmd()

# # rotate
# if job.generate_rotate_list_dir():
#     job.generate_rotate_cmd()
#     job.deploy_rotate_cmd()
 