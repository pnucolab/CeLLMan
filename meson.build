project('cellman', 'cpp',
  version: '0.1.0',
  license: 'AGPL-3.0-or-later',
  meson_version: '>= 0.64.0',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17',
    'warning_level=2',
  ]
)

# Python (meson-python provides the Python module without explicit dependency)
py = import('python').find_installation('python3', pure: false)

# OpenMP (optional)
openmp_dep = dependency('openmp', required: get_option('openmp'))

# numpy include directory
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

cpp_args = []
link_args = []

if openmp_dep.found()
  cpp_args += ['-fopenmp']
  link_args += ['-fopenmp']
endif

# SIMD (AVX-512F) — auto-enabled on x86_64
simd_opt = get_option('simd')
if simd_opt.enabled() or (simd_opt.auto() and host_machine.cpu_family() == 'x86_64')
  cpp_args += ['-mavx512f']
  link_args += ['-mavx512f']
endif

if get_option('buildtype') == 'release'
  cpp_args += ['-O3']
  if get_option('fast-math')
    cpp_args += ['-ffast-math']
  endif
  if get_option('native-arch')
    cpp_args += ['-march=native']
  endif
endif

if get_option('debug-symbols')
  cpp_args += ['-g']
endif

# C++ extension module (compiled from c/utils.cpp; installed as unnamed.utils)
utils_ext = py.extension_module('utils',
  sources: ['cellman/spatial/_core/utils.cpp'],
  include_directories: [
    include_directories(incdir_numpy),
  ],
  dependencies: [openmp_dep],
  cpp_args: cpp_args,
  link_args: link_args,
  install: true,
  subdir: 'cellman/spatial'
)

# Pure-Python package sources
py.install_sources([
  'cellman/__init__.py',
], subdir: 'cellman')

py.install_sources([
  'cellman/spatial/__init__.py',
  'cellman/spatial/_analysis.py',
  'cellman/spatial/_dataset.py',
  'cellman/spatial/_kde_gpu.py',
  'cellman/spatial/_mapping_gpu.py',
  # count-based (blur) backend
  'cellman/spatial/_griddataset.py',
  'cellman/spatial/_panel.py',
  'cellman/spatial/_count.py',
  'cellman/spatial/_tsvm.py',
], subdir: 'cellman/spatial')

# count-based loaders subpackage
py.install_sources([
  'cellman/spatial/_io/__init__.py',
  'cellman/spatial/_io/_grid.py',
  'cellman/spatial/_io/_loaders.py',
], subdir: 'cellman/spatial/_io')
