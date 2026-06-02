import pytest
from pathlib import Path
import yaml

CONFIG_DIR = Path("tests/experiments/embedding/configs/embedding_quality")

ALL_CONFIGS = [
    "transformer_pooling_mean.yaml",
    "transformer_pooling_mean_no_normalization.yaml",
    "transformer_pooling_cls.yaml",
    "transformer_pooling_cls_no_normalization.yaml",
    "transformer_pooling_max.yaml",
    "transformer_pooling_max_no_normalization.yaml",
]


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_valid_config_loads(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert isinstance(config, dict)


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_all_configs_have_required_keys(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert {'experiment_name', 'paths', 'model', 'metrics', 'outputs'}.issubset(config.keys())


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_save_root_is_non_empty_string(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert isinstance(config['paths']['save_root'], str)
    assert len(config['paths']['save_root']) > 0


def test_save_root_created_if_missing(tmp_path):
    out_dir = tmp_path / "new_folder"
    assert not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.exists()


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_model_section_has_required_keys(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    required = {'name', 'pooling', 'normalize', 'device', 'batch_size'}
    assert required.issubset(config['model'].keys())


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_pooling_types(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert config['model']['pooling'] in {'mean', 'cls', 'max'}


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_normalize_is_boolean(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert config['model']['normalize'] in {True, False}


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_batch_size_and_device(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert isinstance(config['model']['batch_size'], int)
    assert config['model']['batch_size'] > 0
    assert isinstance(config['model']['device'], str)
    assert config['model']['device'] in {'cpu', 'cuda'}


@pytest.mark.parametrize("config_file", ALL_CONFIGS)
def test_output_values(config_file):
    config = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert config['outputs']['save_embeddings'] in {True, False}
    assert config['outputs']['save_similarity_matrix'] in {True, False}
