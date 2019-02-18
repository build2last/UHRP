import copy
from .mondrian import mondrian


def mask(data, k, qid_size, relax=False):
	if k <= 1:
		return data, 0
	data = copy.deepcopy(data)
	result, eval_result = mondrian(data, k, relax, qid_size)
	return result, eval_result

def mask_table(table, k, relax=False):
	data = table.qi_values()
	result, eval_result =  mondrian(data, k, relax, table.qid_size)
	return result, eval_result