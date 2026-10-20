def enable_cpu(env):
    from common import SIM, ROOT

    env = dict(env)
    env.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_EGL_DEVICE_ID="0",
        LIBGL_ALWAYS_SOFTWARE="1",
        GALLIUM_DRIVER="llvmpipe",
        LP_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        RC_RENDER_BACKEND="llvmpipe",
        XDG_CACHE_HOME=str(ROOT / "cache/cpu_render"),
        MESA_SHADER_CACHE_DIR=str(ROOT / "cache/cpu_render/mesa"),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(
            SIM / "share/glvnd/egl_vendor.d/50_mesa.json"
        ),
    )
    return env
