
def load_silero_vad(backend='ax650'):
    '''
    backend: ax650, ax630c or onnx
    '''
    package_path = "silero_vad_axera.data"
    model_name = {
        'ax650': 'silero_vad_ax650.axmodel',
        'ax630c': 'silero_vad_ax630c.axmodel',
        'onnx': 'silero_vad.onnx'
    }[backend]

    try:
        import importlib_resources as impresources
        model_file_path = str(impresources.files(package_path).joinpath(model_name))
    except:
        from importlib import resources as impresources
        try:
            with impresources.path(package_path, model_name) as f:
                model_file_path = f
        except:
            model_file_path = str(impresources.files(package_path).joinpath(model_name))

    if backend in ['ax650', 'ax630c']:
        from .SileroAx import SileroAx
        model = SileroAx(str(model_file_path))
    else:
        from .SileroOrt import SileroOrt
        model = SileroOrt(str(model_file_path))

    return model