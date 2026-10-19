import runpy,sys
sys.argv=['scripts/repair_task_environment.py','--instance-id','pvlib__pvlib-python-1048','--numpy-version','1.26.4','--runtime-parent','cs6158/runtime:oh_v0.13.1_jcoza2ihuc7turjp_y766p9z6jyxsud5c','--evidence','results/grade/20261019T220416-957bc845']
runpy.run_path(sys.argv[0],run_name='__main__')
