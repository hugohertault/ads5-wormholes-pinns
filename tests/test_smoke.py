def test_import_main():
    import importlib.util, pathlib
    p = pathlib.Path("main_article_enhanced.py")
    assert p.exists()
    spec = importlib.util.spec_from_file_location("main_article_enhanced", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "run")
