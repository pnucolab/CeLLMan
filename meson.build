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

# Compiler-specific flags. MSVC (cl.exe / clang-cl, the default toolchain on
# native Windows) uses a different flag syntax than GCC/Clang, so we branch on the
# argument syntax and, for good measure, only add flags the compiler accepts
# (get_supported_arguments test-compiles each one). This lets the same tree build
# on Linux/macOS (GCC/Clang) and native Windows (MSVC).
cpp = meson.get_compiler('cpp')
is_msvc = cpp.get_argument_syntax() == 'msvc'

cpp_args = []
link_args = []

# SIMD (AVX-512F) — auto-enabled on x86_64. Unsupported flags are dropped and the
# AVX-512 paths in utils.cpp stay behind their runtime cpuid guard (check_avx512f).
simd_opt = get_option('simd')
if simd_opt.enabled() or (simd_opt.auto() and host_machine.cpu_family() == 'x86_64')
  if is_msvc
    simd_flags = ['/arch:AVX512']
  else
    simd_flags = ['-mavx512f']
  endif
  cpp_args += cpp.get_supported_arguments(simd_flags)
  if not is_msvc
    link_args += cpp.get_supported_link_arguments(simd_flags)
  endif
endif

# Optimization level is set by buildtype (release -> -O3 on GCC/Clang, /O2 on MSVC),
# so we do NOT add an explicit -O3 here (MSVC would reject it). Only opt-in extras:
if get_option('buildtype') == 'release'
  extra_args = []
  if get_option('fast-math')
    if is_msvc
      extra_args += ['/fp:fast']
    else
      extra_args += ['-ffast-math']
    endif
  endif
  if get_option('native-arch') and not is_msvc
    extra_args += ['-march=native']  # no MSVC equivalent
  endif
  cpp_args += cpp.get_supported_arguments(extra_args)
endif

if get_option('debug-symbols')
  if is_msvc
    cpp_args += cpp.get_supported_arguments(['/Z7'])
  else
    cpp_args += cpp.get_supported_arguments(['-g'])
  endif
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
