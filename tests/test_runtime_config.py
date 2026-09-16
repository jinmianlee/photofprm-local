from backend import runtime_config, single_photo


def test_config_paths_are_portable_and_relative_to_project(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_config, 'ROOT', tmp_path)
    (tmp_path/'photoform.json').write_text('{"ai_env":".ai-env","model_dir":"shared/turbo"}')
    c = runtime_config.config()
    assert c['ai_env'] == tmp_path/'.ai-env'
    assert c['model_dir'] == tmp_path/'shared/turbo'
    before = runtime_config.fingerprint(c)
    c['model_dir'].mkdir(parents=True)
    (c['model_dir']/'config.yaml').write_text('changed')
    assert runtime_config.fingerprint(c) != before


def test_invalid_config_gives_actionable_error_without_crashing_system(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_config, 'ROOT', tmp_path)
    (tmp_path/'photoform.json').write_text('{')
    result = single_photo.capability()
    assert result['ready'] is False and 'photoform.json' in result['message']
