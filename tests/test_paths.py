from tess_sso.paths import (
    ENV_VAR,
    asteroid_dir,
    data_root,
    find_project_root,
    observation_dir,
    observation_key,
    reference_dir,
)


def test_explicit_root_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env"))
    assert data_root(tmp_path / "arg") == tmp_path / "arg"


def test_env_var_used_when_no_root(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env"))
    assert data_root() == tmp_path / "env"


def test_default_is_data_tess_sso_in_the_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "notebooks").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(project / "notebooks")
    assert find_project_root() == project.resolve()
    assert data_root() == project.resolve() / "data" / "tess_sso"


def test_existing_data_folder_marks_the_root(tmp_path):
    (tmp_path / "work" / "data").mkdir(parents=True)
    (tmp_path / "work" / "deep" / "er").mkdir(parents=True)
    assert find_project_root(tmp_path / "work" / "deep" / "er") == (tmp_path / "work").resolve()


def test_layout(tmp_path):
    assert observation_key(71, 2, 2) == "s0071-2-2"
    assert asteroid_dir("2000_PH5", tmp_path) == tmp_path / "2000_PH5"
    assert observation_dir("95", 71, 2, 2, tmp_path) == tmp_path / "95" / "s0071-2-2"
    assert reference_dir(tmp_path) == tmp_path / "reference"
