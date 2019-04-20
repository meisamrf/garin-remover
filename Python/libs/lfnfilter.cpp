// lfnfilter python wrapper 
// by Meisam Rakhshanfar

#include <Python.h>

static PyObject *GenError;


class LFNFilter
{
public:
	static void bilinear2(const float * input, float * dataout, int img_row, int img_col);
	static void dftshrink(const float * input, float * dataout, int img_row, int img_col, float nv);
};

PyObject* py_bilinear2(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2;
	Py_buffer b_imgin, b_imgout;

	if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
		return NULL;

	if (PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0)
		return NULL;
	
	if (PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0)
		return NULL;

	if (b_imgin.itemsize != 4 || b_imgout.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgin.ndim != 2 || b_imgout.ndim != 2) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}
	
	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];

	int img_row_o = (int)b_imgout.shape[0];
	int img_col_o = (int)b_imgout.shape[1];

	if (img_row*2 != img_row_o || img_col*2 != img_col_o) {
		PyErr_SetString(GenError, "Output dimension error .\n");
		return NULL;
	}
	
	LFNFilter::bilinear2((float *)b_imgin.buf, (float *)b_imgout.buf, img_row, img_col);

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgin);
	PyBuffer_Release(&b_imgout);

	return res;
}

PyObject* py_dftshrink(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2;
	Py_buffer b_imgin, b_imgout;
	float sigma;

	if (!PyArg_ParseTuple(args, "OOf", &arg1, &arg2, &sigma))
		return NULL;

	if (PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0)
		return NULL;

	if (PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0)
		return NULL;


	if (b_imgin.itemsize != 4 || b_imgout.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgin.ndim != 2 || b_imgout.ndim != 2) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}

	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];

	int img_row_o = (int)b_imgout.shape[0];
	int img_col_o = (int)b_imgout.shape[1];


	if (img_row != img_row_o || img_col != img_col_o) {
		PyErr_SetString(GenError, "Output dimension error .\n");
		return NULL;
	}

	LFNFilter::dftshrink((float *)b_imgin.buf, (float *)b_imgout.buf, img_row, img_col, sigma*sigma);

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgin);
	PyBuffer_Release(&b_imgout);

	return res;
}

PyMethodDef lfnfilterMethods[] = {
	{ "bilinear2",(PyCFunction)py_bilinear2,METH_VARARGS,"bilinear x2 interpolation" },
	{ "dftshrink",(PyCFunction)py_dftshrink,METH_VARARGS,"STFT shrinkage filter" },
	{ NULL, NULL, 0, NULL }
};


static struct PyModuleDef lfnfilter_module = {
	PyModuleDef_HEAD_INIT,
	"lfnfilter",
	"lfnfilter Module C++",
	-1,			  
	lfnfilterMethods
};

PyMODINIT_FUNC
PyInit_lfnfilter(void)
{
	PyObject *m = PyModule_Create(&lfnfilter_module);

	if (m == NULL)
		return NULL;

	GenError = PyErr_NewException("lfnfilter.error", NULL, NULL);
	Py_INCREF(GenError);
	PyModule_AddObject(m, "error", GenError);

	return m;

}
