from distutils.core import setup, Extension
import platform

libname = 'lfnfilter'
if platform.system()=='Darwin':
     libname = 'lfnfilter.mac'

module = Extension('lfnfilter',
                    sources = ['lfnfilter.cpp'],
                    include_dirs = [],
                    libraries = [libname],
                    library_dirs = ['/usr/local/lib', './'],                    
                    extra_compile_args=['-std=c++11'])
 
setup(name = 'lfnfilter',
      version = '1.0',
      description = 'cascaded multi-domain filter tool for fast image denoising',
      ext_modules = [module])
