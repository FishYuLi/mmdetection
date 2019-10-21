import logging
import os.path
from lvis import LVIS, LVISResults, LVISEval

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "data/lvis/lvis_v0.5_val.json"
RESULT_ROOT = './work_dirs/htc_r50_fpn_1x_lvis/'
RESULT_PATH_BBOX = os.path.join(RESULT_ROOT, "results.pkl.bbox.json")
RESULT_PATH_SEGM = os.path.join(RESULT_ROOT, "results.pkl.segm.json")

print('Eval Bbox:')
ANN_TYPE = 'bbox'
lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH_BBOX, ANN_TYPE)
lvis_eval.run()
lvis_eval.print_results()

print('Eval Segm:')
ANN_TYPE = 'segm'
lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH_SEGM, ANN_TYPE)
lvis_eval.run()
lvis_eval.print_results()
